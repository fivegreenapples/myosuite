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
        # rewards being at the target y position which is based on target velociyy and number of steps covered.
        # aims to ensure speed is maintained rather than slowly slipping being
        "gaussian_y_pos": 0,
        # like gaussian_plateau_y_vel but stretches the gaussian so the gradient is not flat at v==0 when target is high (e.g. > 2.5)
        # aims to better support targetting specific running speeds.
        "stretched_gaussian_plateau_y_vel": 0,
        # linear reward up to the target velocity and 1 thereafter.
        # simpler version of above. just simpler without the smooth gradients of a gaussian
        "plateau_y_vel": 0,
    }

    def _setup(
        self,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        x_drift_plateau: float = 0.0,
        curriculum=None,
        print_debug=False,
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

        # used for x_drift cost term
        self._x_drift_plateau = x_drift_plateau
        # used to define a y_vel and y_pos curriculum
        self._curriculum = curriculum
        # used for diagnostics when testing
        self._print_debug = print_debug

        super()._setup(
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )

        # Calculate y_vel curriculum ahead of time
        # Default to incoming target velocity

        # MAX_STEPS is defined when registering the env. Not possible to override this via
        # constructor, and if we want to change seems to require different registrations.
        # Also seems impossible to get the max_episodes_steps from inside the environment
        # owing to how the env is wrapped. So we re-define it here for simplicity.
        MAX_STEPS = 1000

        self._y_vel_curriculum = []
        if self._curriculum:
            # expect a dict with at least a "type" key
            if not isinstance(self._curriculum, dict) or "type" not in self._curriculum:
                raise ValueError("Curriculum must be a dict with a key of 'type' key")

            if self._curriculum["type"] == "random":
                # for random curriculum expect dict of form
                # {
                #     "v_steps": int, .......... how many steps to stay at a particular speed
                #     "v_min": float, .......... minimum velocity to target
                #     "v_max": float, .......... maximum velocity to target
                # }
                # curriculum chooses a new random speed between v_min and v_max every v_steps.
                if (
                    "v_steps" not in self._curriculum
                    or "v_min" not in self._curriculum
                    or "v_max" not in self._curriculum
                ):
                    raise ValueError(
                        "Random curriculum must have 'v_steps', 'v_min' and 'v_max'"
                    )

                v_steps = self._curriculum["v_steps"]
                v_min = self._curriculum["v_min"]
                v_range = self._curriculum["v_max"] - v_min

                for _ in range(0, MAX_STEPS, v_steps):
                    new_target = v_min + (np.random.random() * v_range)
                    self._y_vel_curriculum.extend([new_target] * v_steps)

            elif self._curriculum["type"] == "ramp":
                # for ramp curriculum expect dict of form
                # {
                #     "v_min": float, .......... minimum velocity to target
                #     "v_max": float, .......... maximum velocity to target
                # }
                # curriculum gradually increases speed from v_min to v_max with same
                # delta across all steps

                v_min = self._curriculum["v_min"]
                v_inc = (self._curriculum["v_max"] - v_min) / (MAX_STEPS - 1)

                for idx in range(MAX_STEPS):
                    self._y_vel_curriculum.append(v_min + (idx * v_inc))

            elif self._curriculum["type"] == "stair":
                # for stair curriculum expect dict of form
                # {
                #     "v_min": float, .......... minimum velocity to target
                #     "v_max": float, .......... maximum velocity to target
                #     "v_inc": float, .......... v increase between steps
                # }
                # curriculum gradually increases speed from v_min to v_max with v_inc
                # as target increase between steps

                v_min = self._curriculum["v_min"]
                v_max = self._curriculum["v_max"]
                v_inc = self._curriculum["v_inc"]
                v_range = v_max - v_min
                num_intervals = v_range // v_inc
                final_inc = v_max - (v_min + (v_inc * num_intervals))
                if final_inc > 0.05:
                    num_intervals += 1

                num_stages = num_intervals + 1
                v_steps = MAX_STEPS // num_stages

                if v_steps < 1:
                    # v_inc is too small to have 1 or more steps per stage
                    raise ValueError(
                        "stair curriculum increase too small for number of steps"
                    )

                v_delta = 0
                remaining_steps = MAX_STEPS
                while remaining_steps > 0:
                    stage_steps = min(v_steps, remaining_steps)
                    remaining_steps -= stage_steps

                    target = min(v_max, v_min + v_delta)
                    v_delta += v_inc

                    self._y_vel_curriculum.extend([target] * stage_steps)
            else:
                raise ValueError(
                    f"Unhandled curriculum type: '{self._curriculum['type']}'"
                )
        else:
            self._y_vel_curriculum = [self.target_y_vel] * MAX_STEPS

        assert len(self._y_vel_curriculum) == MAX_STEPS

    def step(self, *args, **kwargs):
        self._prev_ctrl = self.sim.data.ctrl.copy()

        # Update target_y_vel with val from curriculum (the _y_vel_curriculum list is
        # also created for standard constant target velocity for simplicity)
        _prev_vel = self.target_y_vel
        self.target_y_vel = self._y_vel_curriculum[self.steps]

        if self._print_debug and _prev_vel != self.target_y_vel:
            print(
                f"New target y vel: {self.target_y_vel:.2f} m/s"
                f" ({self.target_y_vel*3.6:.1f} kph)"
            )

        return super().step(*args, **kwargs)

    def _plateau_pos(self, p, target, allowance):
        # calculates a distance away from target allowing for a "safe zone"
        # `allowance` is the distance either side of target that gets zero cost.
        # i.e. p is allowed to be target +/- allowance
        return max(0, abs(p - target) - allowance)

    def _gaussian_pos(self, p, target, breadth):
        # calculates a gaussian around the target position
        # breadth controls how wide the curve is and hence how severe the drop off either side
        return np.exp(-np.square((p - target) / breadth))

    def _plateau_vel(self, v, target):
        if target > 0 and v < target:
            return v / target
        return 1.0

    def _gaussian_vel(self, v, target):
        return np.exp(-np.square(v - target))

    def _gaussian_plateau_vel(self, v, target):
        if v < target:
            return np.exp(-np.square(v - target))
        return 1.0

    def _stretched_gaussian_plateau_vel(self, v, target):
        # The stretching ensures there is a reasonable gradient from v==0 up to the target
        # Without it, the gaussian is more or less flat around v==0 when the target is >2.5
        if v < target:
            return np.exp(-np.square(2 * ((v / target) - 1)))
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
        x_pos, y_pos, _ = self._get_com()
        x_vel, y_vel = self._get_com_velocity()

        # frame_skip == 10 - from BaseV0 (same actions applied for 10 frames during step)
        # timestep = 0.001s - from XML
        # dt = 0.01s (time per step)
        SECONDS_PER_STEP = 0.01
        target_y_pos = self.target_y_vel * self.steps * SECONDS_PER_STEP

        gaussian_plateau_y_vel = self._gaussian_plateau_vel(y_vel, self.target_y_vel)

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("x_drift", self._plateau_pos(x_pos, 0, self._x_drift_plateau)),
                ("gaussian_y_pos", self._gaussian_pos(y_pos, target_y_pos, 5)),
                ("y_vel", y_vel),
                ("plateau_y_vel", self._plateau_vel(y_vel, self.target_y_vel)),
                # don't use target_x_vel as this term is only intended to avoid sideways drift
                ("gaussian_x_vel", self._gaussian_vel(x_vel, 0)),
                # provide gaussian_plateau_y_vel for more descriptive label, and gaussian_vel for bw compat
                ("gaussian_plateau_y_vel", gaussian_plateau_y_vel),
                ("gaussian_vel", gaussian_plateau_y_vel),
                (
                    "stretched_gaussian_plateau_y_vel",
                    self._stretched_gaussian_plateau_vel(y_vel, self.target_y_vel),
                ),
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

    # Override get_obs_dict to insert vals for target y velocity if a curriculum is set
    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        if not self._curriculum:
            # preserve compatability with use of this environment when no curriculum is set
            return obs_dict

        new_obs = {}
        for k in obs_dict:
            if k == "act":
                # Insert target velocity observations before "act"
                # "act" must stay at end of dict to satisfy expectations of the custom
                # replay buffer AdaptiveEnergyBuffer used in depRL.
                #
                # We supply the target and difference from target. Possibly these are
                # redundant but perhaps this makes it easier for the learning process.
                _, y_vel = self._get_com_velocity().copy()
                new_obs["target_vel"] = np.array(
                    [
                        self.target_y_vel,  # the actual target
                        y_vel - self.target_y_vel,  # difference from target
                    ]
                )
            new_obs[k] = obs_dict[k]

        return new_obs
