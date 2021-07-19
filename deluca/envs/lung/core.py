# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from deluca.envs.core import Env


class Lung(Env):
    """
    Todos:
        - Plot incrementally
        - Save fig as fig.self
        - Modify close to delete fig and accumulated data
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    # scale parameter controls the relative size of the radius compared to base
    def render(self, mode="human", scale=1.0):
        screen_width = 600
        screen_height = 400

        # world_width = self.x_threshold * 2
        # scale = screen_width / world_width
        bottom_rect_y = 100  # TOP OF bottom_rect
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        bottom_width = 250.0
        bottom_height = 5.0
        top_width = 250.0
        top_height = 5.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            # set up bottom rect
            l, r, t, b = -bottom_width / 2, bottom_width / 2, bottom_height / 2, -bottom_height / 2
            # axleoffset = cartheight / 4.0
            bottom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.bottom_trans = rendering.Transform()
            bottom.add_attr(self.bottom_trans)
            self.viewer.add_geom(bottom)

            # set up top rect
            l, r, t, b = 0, top_width, top_height / 2, -top_height / 2
            # axleoffset = cartheight / 4.0
            top = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.top_trans = rendering.Transform(translation=(-bottom_width / 2, 0))
            top.add_attr(self.top_trans)
            top.add_attr(self.bottom_trans)
            self.viewer.add_geom(top)

            # set up balloon circle
            balloon = rendering.make_circle(radius=20, res=30, filled=False)
            self.balloon_trans = rendering.Transform()
            balloon.add_attr(self.balloon_trans)
            balloon.add_attr(self.bottom_trans)
            self.viewer.add_geom(balloon)

        if self.state is None:
            return None

        # translate bottom, which will automatically translate the top and balloon
        self.bottom_trans.set_translation(screen_width / 2.0, bottom_rect_y)

        # Edit the balloon and top
        if "pressure" in self.state.keys():
            pres = self.state["pressure"]
        elif "normalized_pressures" in self.state.keys():
            pres = self.pressure
        print("pres:" + str(pres))
        radius_polynomial = np.array(
            [0.01 * pres, -1, 0, 0, 0, 0, 0, 1]
        )  # https://en.wikipedia.org/wiki/Two-balloon_experiment
        possible_radii = np.roots(radius_polynomial)
        print("possible_radii:" + str(possible_radii))
        real_possible_radii = [r for r in possible_radii if (not np.iscomplex(r) and r > 0)]
        print("real_possible_radii:" + str(real_possible_radii))
        # assert(len(real_possible_radii) == 1)
        # radius = real_possible_radii[0]
        radius = max(real_possible_radii) * scale
        print("radius:" + str(radius))
        self.balloon_trans.set_translation(0.6 * bottom_width / 2.0, 0.1 * 20 * radius)
        self.balloon_trans.set_scale(0.1 * radius, 0.1 * radius)

        num = 0.1 * 20 * radius
        den = ((0.8 * bottom_width) ** 2 + (0.1 * 20 * radius) ** 2) ** 0.5
        top_theta = 2 * np.arcsin(num / den)
        self.top_trans.set_rotation(top_theta)

        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
