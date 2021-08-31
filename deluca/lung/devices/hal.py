""" Module for interacting with physical and/or simulated devices installed on the ventilator.

"""

import os
from importlib import import_module
from ast import literal_eval
from deluca.lung.devices.sensors import Sensor
from deluca.lung.devices.base import PigpioConnection

import deluca.lung.devices.valves as valves
import configparser


default_hal_config = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config/devices.ini")


class Hal:
    """Hardware Abstraction Layer for ventilator hardware.
    Defines a common API for interacting with the sensors & actuators on the ventilator. The types of devices installed
    on the ventilator (real or simulated) are specified in a configuration file.
    """

    def __init__(self, host=None, config_file=default_hal_config):
        """Initializes HAL from config_file.
            For each section in config_file, imports the class <type> from module <module>, and sets attribute
            self.<section> = <type>(**opts), where opts is a dict containing all of the options in <section> that are
            not <type> or <section>. For example, upon encountering the following entry in config_file.ini:

                [adc]
                type   = ADS1115
                module = devices
                i2c_address = 0x48
                i2c_bus = 1

            The Hal will:
                1) import deluca.lung.devices.ADS1115 (or ADS1015) as a local variable:
                        class_ = getattr(import_module('.devices', 'vent.io'), 'ADS1115')

                2) Instantiate an ADS1115 object with the arguments defined in config_file and set it as an attribute:
                        self._adc = class_(gpio=self._gpio,address=0x48,i2c_bus=1)

            Note: RawConfigParser.optionxform() is overloaded here s.t. options are case sensitive (they are by default
            case insensitive). This is necessary due to the kwarg MUX which is so named for consistency with the config
            registry documentation in the ADS1115 datasheet. For example, A P4vMini pressure_sensor on pin A0 (MUX=0)
            of the ADC is passed arguments like:

            analog_sensor = AnalogSensor(
                _ig=self._gpio,
                adc=self._adc,
                MUX=0,
                offset_voltage=0.25,
                output_span = 4.0,
                conversion_factor=2.54*20
            )

            Note: ast.literal_eval(opt) interprets integers, 0xFF, (a, b) etc. correctly. It does not interpret strings
            correctly, nor does it know 'adc' -> self._adc; therefore, these special cases are explicitly handled.
        Args:
            config_file (str): Path to the configuration file containing the definitions of specific components on the
                ventilator machine. (e.g., config_file = "vent/io/config/devices.ini")
        """
        self._setpoint_in = 0.0  # setpoint for inspiratory side
        self._setpoint_ex = 0.0  # setpoint for expiratory side
        self._adc = object
        self._control_valve = object
        self._expiratory_valve = object
        self._pressure_sensor = object
        self._flow_sensor_ex = object
        self._gpio = object
        self.config = configparser.RawConfigParser()
        self.config.optionxform = lambda option: option
        self.config.read(config_file)

        for section in self.config.sections():
            sdict = dict(self.config[section])
            class_ = getattr(import_module("." + sdict["module"], "deluca.lung"), sdict["type"])
            opts = {
                key: sdict[key]
                for key in sdict.keys()
                - (
                    "module",
                    "type",
                )
            }
            for key in opts.keys():
                if key == "adc":
                    opts[key] = self._adc
                else:
                    opts[key] = literal_eval(opts[key])
            if section == "gpio":
                if host is not None:
                    opts["host"] = host
                setattr(self, "_gpio", class_(**opts))
                # Clean up i2c
                for i in range(32):
                    try:
                        self._gpio.i2c_close(i)
                    except:
                        continue
            else:
                setattr(self, "_" + section, class_(gpio=self._gpio, **opts))

            print(
                "  [ {device_name:^19} ]  opts: {device_options}".format(
                    device_name=section, device_options=opts
                )
            )  # debug

    @property
    def pressure(self) -> float:
        """Returns the pressure from the primary pressure sensor."""
        return self._pressure_sensor.get()

    @property
    def flow_ex(self) -> float:
        """ The measured flow rate expiratory side."""
        return self._flow_sensor_ex.get()

    @property
    def setpoint_in(self) -> float:
        """ The currently requested flow for the inspiratory proportional control valve as a proportion of maximum."""
        return self._setpoint_in

    @setpoint_in.setter
    def setpoint_in(self, value: float):
        """Sets the openness of the inspiratory valve to the requested value.

        Args:
            value: Requested flow, as a proportion of maximum. Must be in [0, 1].
        """
        if not 0 <= value <= 100:
            raise ValueError("setpoint must be a number between 0 and 100")
        self._control_valve.setpoint = value
        self._setpoint_in = value

    @property
    def setpoint_ex(self) -> float:
        """ The currently requested flow on the expiratory side as a proportion of the maximum."""
        return self._setpoint_ex

    @setpoint_ex.setter
    def setpoint_ex(self, value):
        """Sets the openness of the expiratory valve to the requested value.

        Args:
            value (float): Requested flow, as a proportion of maximum. Must be either 0 or 1 for OnOffValve, and between
                0 and 1 for a (proportional) control valve.
        """
        if isinstance(self._expiratory_valve, valves.OnOffValve):
            if value not in (0, 1):
                raise ValueError("setpoint must be either 0 or 1 for an On/Off expiratory valve")
            elif value == 1:
                self._expiratory_valve.open()
            else:
                self._expiratory_valve.close()
        elif isinstance(self._expiratory_valve, valves.PWMControlValve):
            if not 0 <= value <= 100:
                raise ValueError(
                    "setpoint must be between 0 and 100 for an expiratory control valve"
                )
            else:
                self._expiratory_valve.setpoint = value
        self._setpoint_ex = value
