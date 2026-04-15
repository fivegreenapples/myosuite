"""=================================================
# Copyright (c) Ben Holland
Authors  :: Ben Holland (ben@fivegreenapples.com)
=================================================

Implementation of the Walker environment using the reward function developed in
Schumacher et al. 2025
Emergence of natural and robust bipedal walking by learning from biologically plausible objectives
https://doi.org/10.1016/j.isci.2025.112203

Heavily inspired by the SCONE Gym implementation:
https://github.com/tgeijten/sconegym/blob/main/sconegym/gaitgym.py
"""

import collections

import mujoco
import numpy as np

from .walk_v0 import WalkEnvV0


class NaturalAndRobustWalker(WalkEnvV0):
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        # These weights are taken from the sconegym implementation.
        # All but gaussian_vel are mentioned in the paper and do indeed match sconegym.
        # Interestingly, they are rounded to lower s.f. in the paper.
        "gaussian_vel": 10,
        "grf": -0.07281,
        "smooth_exc": -0.097,
        "number_muscles": -1.57929,
        "joint_limit": -0.1307,
    }

    def _setup(
        self,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **kwargs,
    ):
        super()._setup(
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )

    def step(self, *args, **kwargs):
        self._prev_ctrl = self.sim.data.ctrl.copy()
        return super().step(*args, **kwargs)

    def _y_vel(self):
        _, y_vel = self._get_com_velocity()
        return y_vel

    def _gaussian_plateau_vel(self):
        _, y_vel = self._get_com_velocity()

        if y_vel < self.target_y_vel:
            return np.exp(-np.square(y_vel - self.target_y_vel))

        return 1.0

    def _grf(self):
        r_grf = (
            self.sim.data.sensor("r_foot").data[0]
            + self.sim.data.sensor("r_toes").data[0]
        )
        l_grf = (
            self.sim.data.sensor("l_foot").data[0]
            + self.sim.data.sensor("l_toes").data[0]
        )
        weight = 9.8 * sum(self.sim.model.body_mass)
        # The feet and toe sensors are <touch> sensors which return a single scalar value
        # for surface forces acting through the touch "site" along a normal to the
        # contacting surface. At least I think that's what they do.
        # Either way, the values are in Newtons. We normalized this against the weight
        # so the normalized_grf is in units of body weight "BW" (this mirrors how Scone
        # returns contact_load)
        normalized_grf = (r_grf + l_grf) / weight
        # and then return this value clipped below 1.2 - a magic number from the original
        # paper which serves to avoid any penalty for grfs which would occur in normal
        # walking.
        return max(0, normalized_grf - 1.2)

    def _exc_smooth_cost(self):
        # ctrl is the excitation array
        # act is the resulting activation state
        # actuator_force is the resulting force
        delta_excs = self.sim.data.ctrl - self._prev_ctrl
        return np.mean(np.square(delta_excs))

    def _number_muscle_cost(self):
        # 0.15 is a magic number from the paper.
        return self._get_proportion_active_muscles(0.15)

    def _get_proportion_active_muscles(self, threshold):
        # Gets the proportion of muscles whose activations are above a threshold.
        return np.count_nonzero(self.sim.data.act > threshold) / self.sim.data.act.size

    def _joint_limit_torques(self):
        # Use the efc arrays directly. These contain the details of the currently active
        # constraints. I think. We need to do this to extract the joint limit constraints
        # and thus the forces/torques imposed by those constraints.
        #
        # Have found that the number of joint limit constraints this finds is always the
        # same number of joints that are out of their defined ranges. Which is encouraging.
        #
        # Use efc_type array to select the constraints that are joint-limits. And use
        # that to select from the actual efc_force array. However, this will return
        # values in N for slide joints and Nm for hinge joints. So summing them is not
        # technically a valid thing to do, and as far as I can tell, the Scone
        # implementation only cares about torques. So, we can use the efc_id array to
        # find the joint ids, use that to index the jnt_type array, and disambiguate the
        # hinges from the slides.
        #
        # For now we only deal with the torques (from hinge joints) per the Scone
        # implementation. Adding support for slide limit forces is left as a todo.
        joint_limit_constraint_idx = np.nonzero(
            self.sim.data.efc_type == self.sim.lib.mjtConstraint.mjCNSTR_LIMIT_JOINT
        )
        jnt_forces = self.sim.data.efc_force[joint_limit_constraint_idx]
        jnt_ids = self.sim.data.efc_id[joint_limit_constraint_idx]
        jnt_types = self.sim.model.jnt_type[jnt_ids]
        sum_hinge_torques = np.sum(
            jnt_forces[np.nonzero(jnt_types == self.sim.lib.mjtJoint.mjJNT_HINGE)]
        )

        # Scone implementation returns a mean average across all axes and all
        # joints. Which, I think in MuJoCo land means divide by the number of hinge
        # joints as each hinge in MuJoCo only has one axis (in Scone it looks like a
        # single joint incorporates all 3 axes).
        num_hinge_joints = np.count_nonzero(
            self.sim.model.jnt_type == self.sim.lib.mjtJoint.mjJNT_HINGE
        )

        return sum_hinge_torques / num_hinge_joints

    def _self_contact_cost(self):
        # Sum of all contact force magnitudes between bodies in the model.
        total_force = 0.0
        floor_geom_id = self.sim.model.geom_name2id("floor")

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = contact.geom[0]
            geom2 = contact.geom[1]

            # Skip contacts involving the floor geom
            if geom1 == floor_geom_id or geom2 == floor_geom_id:
                continue

            dims = contact.dim
            efc_start = contact.efc_address
            force = self.sim.data.efc_force[efc_start : efc_start + dims]
            print(i, force)

            # # Use just the normal + tangential force magnitude
            # total_force += np.linalg.norm(force[:3])

        return total_force

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("y_vel", self._y_vel()),
                ("gaussian_vel", self._gaussian_plateau_vel()),
                ("grf", self._grf()),
                ("smooth_exc", self._exc_smooth_cost()),
                ("number_muscles", self._number_muscle_cost()),
                ("joint_limit", self._joint_limit_torques()),
                ("self_contact_cost", self._self_contact_cost()),
                # Must keys
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()],
            axis=0,
        )
        return rwd_dict
