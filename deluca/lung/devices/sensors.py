from deluca.lung.devices import I2CDevice, be16_to_native
from abc import ABC, abstractmethod
import random
from collections import deque

import time
import numpy as np


class Sensor(ABC):
    """Abstract base Class describing generalized sensors. Defines a mechanism for limited internal storage of recent
    observations and methods to pull that data out for external use.
    """

    _DEFAULT_STORED_OBSERVATIONS = 128

    def __init__(self):
        """ Upon creation, calls update() to ensure that if get is called there will be something to return."""
        self._maxlen_data = self._DEFAULT_STORED_OBSERVATIONS
        self._data = {
            "timestamp": deque(maxlen=self.maxlen_data),
            "value": deque(maxlen=self.maxlen_data),
        }

    def update(self) -> float:
        """Make a sensor reading, verify that it makes sense and store the result internally. Returns True if reading
        was verified and False if something went wrong.
        """
        value = self._read()
        self._data["timestamp"].append(time.time())
        self._data["value"].append(value)
        return self._verify(value)

    def get(self, average=False) -> float:
        """ Return the most recent sensor reading, or an average of readings since last get(). Clears internal memory so as not to have stale data."""
        # FIXME - timeout decorators for everyone
        if len(self._data["value"]) == 0:
            self.update()
        if average:
            value = np.mean(self._data["value"])
        else:
            value = self._data["value"].pop()
        self._clear()
        return value

    def age(self) -> float:
        """ Returns the time in seconds since the last sensor update, or -1 if never updated."""
        if not self._data["timestamp"]:
            return -1.0
        else:
            return time.time() - self._data["timestamp"][-1]

    def reset(self):
        """Resets the sensors internal memory. May be overloaded by subclasses to extend functionality specific to a
        device.
        """
        self._clear()

    def _clear(self):
        """ Resets the sensors internal memory. """
        for key in self._data.keys():
            self._data[key].clear()

    @property
    def data(self) -> np.array:
        """Returns all Locally-stored observations.

        Returns:
            np.array: An array of timestamped observations arranged oldest to newest.
        """
        result = np.column_stack([self._data["timestamp"], self._data["value"]])
        self._clear()
        return result

    @property
    def maxlen_data(self) -> int:
        """Returns the number of observations kept in the Sensor's internal ndarray. Once the ndarray has been filled,
        the sensor begins overwriting the oldest elements of the ndarray with new observations such that the size of the
        internal storage stays constant.
        """
        return self._maxlen_data

    @maxlen_data.setter
    def maxlen_data(self, new_data_length):
        """Set a new length for stored observations. Clears existing
        observations and resets.

        Args:
            new_data_length (int): The new length of internal observation storage
        """
        if type(new_data_length) != int:
            raise ValueError
        new_data = {
            "value": deque(maxlen=new_data_length),
            "timestamp": deque(maxlen=new_data_length),
        }
        for key in new_data.keys():
            new_data[key].extend(self._data[key])
            self._data[key].clear()
            self._data[key] = new_data[key]
        self._maxlen_data = new_data_length

    def _read(self) -> float:
        """ Calls _raw_read and scales the result before returning it."""
        return self._convert(self._raw_read())

    @abstractmethod
    def _verify(self, value):
        """Validate reading and throw exception/alarm if sensor does not appear to be working correctly."""

    @abstractmethod
    def _convert(self, raw):
        """Converts a raw reading from a sensor in whatever format the device communicates with into a meaningful
        result.
        """

    @abstractmethod
    def _raw_read(self):
        """Requests a new observation from the device and returns the raw result in whatever format/units the device
        communicates with.
        """


class AnalogSensor(Sensor):
    """Generalized class describing an analog sensor attached to the ADS1115 analog to digital converter. Inherits from
    the sensor base class and extends with functionality specific to analog sensors attached to the ADS1115. If
    instantiated without a subclass, conceptually represents a voltmeter with a normalized output.
    """

    # TODO The offset voltage/output span & verify/calibrate stuff needs a rethink.
    _DEFAULT_offset_voltage = 0
    _DEFAULT_output_span = 5
    _DEFAULT_CALIBRATION = {
        "offset_voltage": _DEFAULT_offset_voltage,
        "output_span": _DEFAULT_output_span,
        "conversion_factor": 1,
    }

    def __init__(self, adc, **kwargs):
        """Links analog sensor on the ADC with configuration options specified. If no options are specified, it assumes
        the settings currently on the ADC.

        Args:
            adc (vent.io.devices.ADS1115): The adc object to which the AnalogSensor is attached
            **kwargs: `field=value` - see vent.io.devices.ADS1115 for additional documentation. Strongly suggested to
                specify `MUX=adc_pin` here unless you know what you're doing.
        """
        super().__init__()
        self.adc = adc
        if "MUX" not in (kwargs.keys()):
            raise TypeError("User must specify MUX for AnalogSensor creation")
        kwargs = {key: kwargs[key] for key in kwargs.keys() - ("gpio",)}
        self._check_and_set_attr(**kwargs)

    def calibrate(self, **kwargs):
        """Sets the calibration of the sensor, either to the values contained in the passed tuple or by some routine;
        the current routine is pretty rudimentary and only calibrates offset voltage.

        Args:
            **kwargs: calibration_field=value, where calibration field is one of the following: 'offset_voltage',
                output_span' or 'conversion_factor'
        """
        # FIXME
        if kwargs:
            for fld, val in kwargs.items():
                if fld in self._DEFAULT_CALIBRATION.keys():
                    setattr(self, fld, val)
        else:
            for _ in range(50):
                self.update()
                # PRINT FOR DEBUG / HARDWARE TESTING
                print(
                    "Analog Sensor Calibration @ {:6.4f}".format(self.data[self.data.shape[0] - 1]),
                    end="\r",
                )
                time.sleep(0.1)
            self.offset_voltage = np.min(self.data[-50:])
            # PRINT FOR DEBUG / HARDWARE TESTING
            print("Calibrated low-end of AnalogSensor @", " %6.4f V" % self.offset_voltage)

    def _read(self) -> float:
        """ Returns a value in the range of 0 - 1 corresponding to a fraction of the full input range of the sensor."""
        return self._convert(self._raw_read())

    def _verify(self, value) -> bool:
        """Checks to make sure sensor reading was indeed in [0, 1].

        Args:
            value (float): Sensor reading to validate
        """
        report = bool(-1 <= value / self.conversion_factor <= 1)
        # if not report:
        # FIXME: Right now this just expands the calibration range whenever bounds are exceeded, because we're not
        #  familiar enough with our sensors to know when we should really be rejecting values. This approach should
        #  work for debugging/R&D purposes but really ought to be thought through and/or replaced for production.
        #  For example, negative voltages are probably bad. voltages about VDD (~5V) are also probably bad. There is
        #  some expected drift around offset voltage and output span, however, and that drift is going to change
        #  depending on the sensor in question; i.e., voltages between offset_voltage and zero may or may not be ok,
        #  and voltages above the offset+span that do not exceed VDD may or may not be ok as well.
        #    self.offset_voltage = min(self.offset_voltage, value)
        #    self.output_span = max(self.output_span, value - self.offset_voltage)
        #    print('Warning: AnalogSensor calibration adjusted')
        return report

    def _convert(self, raw) -> float:
        """Scales raw voltage into the range 0 - 1.

        Args:
            raw (float): The raw sensor reading to convert.
        """
        return self.conversion_factor * (
            (raw - getattr(self, "offset_voltage")) / getattr(self, "output_span")
        )

    def _raw_read(self) -> float:
        """Builds kwargs from configured fields to pass along to adc, then calls adc.read_conversion(), which returns
        a raw voltage.
        """
        fields = self.adc.USER_CONFIGURABLE_FIELDS
        kwargs = dict(zip(fields, (getattr(self, field) for field in fields)))
        return self.adc.read_conversion(**kwargs)

    def _fill_attr(self):
        """Examines self to see if there are any fields identified as user configurable or calibration that have not
        been write (i.e. were not passed to __init__ as **kwargs). If a field is missing, grabs the default value either
        from the ADC or from _DEFAULT_CALIBRATION and sets it as an attribute.
        """
        for cfld in self.adc.USER_CONFIGURABLE_FIELDS:
            if not hasattr(self, cfld):
                setattr(self, cfld, getattr(self.adc.config, cfld).unpack(self.adc.cfg))
        for dcal, value in self._DEFAULT_CALIBRATION.items():
            if not hasattr(self, dcal):
                setattr(self, dcal, value)

    def _check_and_set_attr(self, **kwargs):
        """Checks to see if arguments passed to __init__ are recognized as user configurable or calibration fields. If
        so, write the value as an attribute like: self.KEY = VALUE. Keeps track of how many attributes are write in this
        way; if at the end there unknown arguments leftover, raises a TypeError; otherwise, calls _fill_attr() to fill
        in fields that were not passed as arguments.

        Args:
            **kwargs: `field=value` - see vent.io.devices.ADS1115 for additional documentation
        """
        allowed = (
            *self.adc.USER_CONFIGURABLE_FIELDS,
            *self._DEFAULT_CALIBRATION.keys(),
        )
        result = 0
        for fld, val in kwargs.items():
            if fld in allowed:
                setattr(self, fld, val)
                result += 1
        if result != len(kwargs):
            raise TypeError("AnalogSensor was passed unknown field(s)", kwargs.items(), allowed)
        self._fill_attr()


class SFM3200(Sensor, I2CDevice):
    """I2C Inspiratory flow sensor manufactured by Sensirion AG. Range: +/- 250 SLM
    Datasheet:
         https://www.sensirion.com/fileadmin/user_upload/customers/sensirion/Dokumente/ ...
            ... 5_Mass_Flow_Meters/Datasheets/Sensirion_Mass_Flow_Meters_SFM3200_Datasheet.pdf
    """

    _DEFAULT_ADDRESS = 0x40
    _FLOW_OFFSET = 32768
    _FLOW_SCALE_FACTOR = 120

    def __init__(self, address=_DEFAULT_ADDRESS, i2c_bus=1, gpio=None):
        """
        Args:
            address (int): The I2C Address of the SFM3200 (usually 0x40)
            i2c_bus (int): The I2C Bus to use (usually `1` on the Raspberry Pi)
            gpio (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        I2CDevice.__init__(self, address, i2c_bus, gpio)
        Sensor.__init__(self)
        self.reset()
        self._start()

    def reset(self):
        """ Extended to add device specific behavior: Asks the sensor to perform a soft reset. 80 ms soft reset time."""
        super().reset()
        self.write_device(0x2000)
        time.sleep(0.08)  # TODO: this should be an await

    def _start(self):
        """Device specific:Sends the 'start measurement' command to the sensor. Start-up time once command has been
        recieved is 'less than 100ms'
        """
        self.write_device(0x1000)
        time.sleep(0.1)  # TODO: this should be an await

    def _verify(self, value) -> bool:
        """No further verification needed for this sensor. Onboard chip handles all that. Could throw in a CRC8 checker
        instead of discarding them in _convert().

        Args:
            value (float): The sensor reading to verify.
        """
        return True

    def _convert(self, raw) -> float:
        """Overloaded to replace with device-specific protocol. Convert raw int to a flow reading having type float
        with units slm. Implementation differs from parent for clarity and consistency with source material.

        Source:
          https://www.sensirion.com/fileadmin/user_upload/customers/sensirion/Dokumente/ ...
            ... 5_Mass_Flow_Meters/Application_Notes/Sensirion_Mass_Flo_Meters_SFM3xxx_I2C_Functional_Description.pdf

        Args:
            raw (int): The raw value read from the SFM3200
        """
        return (raw - self._FLOW_OFFSET) / self._FLOW_SCALE_FACTOR

    def _raw_read(self) -> int:
        """Performs an read on the sensor, converts received bytearray, discards the last two bytes (crc values - could
        implement in future), and returns a signed int converted from the big endian two complement that remains.
        """
        return be16_to_native(self.read_device(4))


class DLiteSensor(AnalogSensor):
    """D-Lite flow sensor setup.
    This consists of the GE D-Lite sensor configured with
    vacuum lines running to an analog differential pressure sensor.
    """

    def __init__(self, adc, **kwargs):

        super().__init__(adc, **kwargs)

    def _convert(self, raw) -> float:
        """Converts the raw differential voltage signal to
        a measurement of flow in liters-per-minute (LPM).

        We calibrate the D-Lite flow readings using the
        (pre-calibrated) Sensirion flow sensor (see SFM3200).
        Args:
            raw (float): The raw sensor reading to convert.
        """
        raw = super()._convert(raw)
        if raw >= 0:
            # converted_flow = (-1.0*np.sqrt(raw)/np.sqrt(fit_param))
            converted_flow = 192.6426 * (raw) ** (1 / 1.9128)
        else:
            converted_flow = -192.6426 * (np.abs(raw)) ** (1 / 1.9128)
        return converted_flow

    def calibrate(self, **kwargs):
        """Do not run a calibration routine.
        Overrides attempt to calibrate.
        """
        return