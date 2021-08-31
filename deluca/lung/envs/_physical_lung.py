# Copyright 2021 The Deluca Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import datetime
import numpy as np
import dataclasses

import deluca.core
from deluca.lung.devices.hal import Hal

class PhysicalLungObservation(deluca.Obj):
    pressure: float = 0.
    flow: float = 0.
    time: float = 0.
    dt: float = 0.
    steps: int = 0
    breaths: int = 0
    prev_time: float = 0.
    start: float = datetime.datetime.now().timestamp()


class PhysicalLung:
    def __init__(
        self,
        host=None,
        sleep=3.0,
        abort=50,
        PEEP=5,
    ):
        hal = Hal(host=host)
        self.hal = hal
        self.sleep = sleep

        self.abort = abort
        self.PEEP = PEEP

        self.breaths = 0
        self.__time = 0.
        self.prev_time = 0.

    def init(self):
        return PhysicalLungObservation(pressure=self.pressure, flow=self.flow)

    @classmethod
    def create(cls, host=None, sleep=3.0, abort=70, PEEP=5):
        args = locals()
        del args["cls"]
        return cls(**args)

    @property
    def pressure(self):
        return self.hal.pressure

    @property
    def flow(self):
        return self.hal.flow_ex

    def wait(self, state, duration):
        time.sleep(duration)

        return dataclasses.replace(state, time=time.time() - state.start)

    def __call__(self, state, action):
        u_in, u_out = action
        if u_out == 1:
            self.breaths += 1
        self.hal.setpoint_in = u_in
        self.hal.setpoint_ex = u_out

        curr_time = time.time() - state.start
        
        new_state = dataclasses.replace(
            state,
            pressure=self.pressure,
            flow=self.flow,
            time=curr_time,
            prev_time=state.time,
            steps=state.steps + 1,
            dt=curr_time - state.time
        )

        if u_out == 1:
            new_state = dataclasses.replace(
                new_state,
                breaths=state.breaths + 1
            )

        return new_state, new_state

    def should_abort(self):
        # timestamp = self.__time
        if self.pressure > self.abort:
            print(f"Pressure of {self.pressure} > {self.abort}; quitting")
            return True

        # self.dt_window = np.roll(self.dt_window, 1)
        # self.dt_window[0] = self.dt

        # if np.mean(self.dt_window) > self.dt_threshold:
            # print(
            # f"dt averaged {100 * self.dt_threshold:.1f}% higher over the last {self.dt_patience} timesteps; quitting"
            # )
            # return False

        # if self.breaths > self.peep_breaths:
            # self.pressure_window = np.roll(self.pressure_window, 1)
            # self.pressure_window[0] = self.pressure

        #if np.mean(self.pressure_window) < self.PEEP * self.peep_threshold:
        #    print("Pressure drop, did you blow up?")
        #    return True

        return False

    def cleanup(self):
        self.hal.setpoint_in = 0
        self.hal.setpoint_ex = 1
        time.sleep(self.sleep)
        self.hal.setpoint_ex = 0
