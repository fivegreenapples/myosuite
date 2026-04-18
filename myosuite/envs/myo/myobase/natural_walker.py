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
        # y_vel is not mentioned in scone implementation but used here in the max speed
        # running roll outs, and set to 1 to have reward == y_vel
        "y_vel": 0,
        # self_contact is not used in walking but used in max speed running. weight is
        # -10 from paper and sconegym.
        "self_contact": 0,
        ##
        ##
        ## Further terms have been added for reward shaping beyond what's in the paper
        ##
        # gaussian_plateau_y_vel is just a more descriptive name for the above gaussian_vel
        "gaussian_plateau_y_vel": 0,
        # gaussian_x_vel is a true symmetric (non-plateau) gaussian for targetting zero x_vel
        "gaussian_x_vel": 0,
        # x_drift is a cost term to penalise moving away from the running centerline
        "x_drift": 0,
    }

    def _setup(
        self,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **kwargs,
    ):
        # pre calculate model weight for grf cost
        self._model_weight = 9.8 * sum(self.sim.model.body_mass)
        # pre calculate number of hinge joints for joint_limit cost
        self._num_hinge_joints = np.count_nonzero(
            self.sim.model.jnt_type == self.sim.lib.mjtJoint.mjJNT_HINGE
        )
        # set floor geom id for self_contact cost
        self._floor_geom_id = self.sim.model.geom_name2id("floor")

        super()._setup(
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )

    def step(self, *args, **kwargs):
        self._prev_ctrl = self.sim.data.ctrl.copy()
        return super().step(*args, **kwargs)

    def _gaussian_vel(self, v, target):
        return np.exp(-np.square(v - target))

    def _gaussian_plateau_vel(self, v, target):
        if v < target:
            return np.exp(-np.square(v - target))

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
        # The feet and toe sensors are <touch> sensors which return a single scalar value
        # for surface forces acting through the touch "site" along a normal to the
        # contacting surface. At least I think that's what they do.
        # Either way, the values are in Newtons. We normalized this against the weight
        # so the normalized_grf is in units of body weight "BW" (this mirrors how Scone
        # returns contact_load)
        normalized_grf = (r_grf + l_grf) / self._model_weight
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
        return sum_hinge_torques / self._num_hinge_joints

    def _self_contact_cost(self):
        # Sum of all contact force magnitudes between bodies in the model.
        total_force = 0.0

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = contact.geom[0]
            geom2 = contact.geom[1]

            # Skip contacts involving the ground
            if geom1 == self._floor_geom_id or geom2 == self._floor_geom_id:
                continue

            # Only worry about the normal force which will be the first force in the list
            # i.e. don't worry about the number of dimensions (contact.dim)
            # Take the absolute value
            force = abs(self.sim.data.efc_force[contact.efc_address])

            total_force += force

        # Now clip to 100 and normalize by 100 so we're in the range [0,1]
        # From the paper this means "only strong and potentially painful self-contacts
        # are considered, while weaker collisions can be safely ignored by the learner."
        total_force = min(total_force, 100)
        total_force /= 100

        return total_force

    def get_reward_dict(self, obs_dict):
        x_pos, _, _ = self._get_com()
        x_vel, y_vel = self._get_com_velocity()

        gaussian_plateau_y_vel = self._gaussian_plateau_vel(y_vel, self.target_y_vel)

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("x_drift", abs(x_pos)),
                ("y_vel", y_vel),
                # don't use target_x_vel as this term is only intended to avoid sideways drift
                ("gaussian_x_vel", self._gaussian_vel(x_vel, 0)),
                # provide gaussian_plateau_y_vel for more descriptive label, and gaussian_vel for bw compat
                ("gaussian_plateau_y_vel", gaussian_plateau_y_vel),
                ("gaussian_vel", gaussian_plateau_y_vel),
                ("grf", self._grf()),
                ("smooth_exc", self._exc_smooth_cost()),
                ("number_muscles", self._number_muscle_cost()),
                ("joint_limit", self._joint_limit_torques()),
                ("self_contact", self._self_contact_cost()),
                # Must keys
                ("sparse", 1),
                ("solved", False),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()],
            axis=0,
        )
        return rwd_dict
