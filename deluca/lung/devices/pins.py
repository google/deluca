from deluca.lung.devices import IODeviceBase
from deluca.lung.devices.base import gpio_command


class Pin(IODeviceBase):
    """ Base Class wrapping pigpio methods for interacting with GPIO pins on
    the raspberry pi. Subclasses include InputPin, OutputPin; along with
    any specialized pins or specific devices defined in vent.io.actuators
    & vent.io.sensors (note: actuators and sensors do not need to be tied
    to a GPIO pin and may instead be interfaced through an ADC or I2C).

    This is an abstract base class. The subclasses InputPin and
    OutputPin extend Pin into a usable form.
    """
    _PIGPIO_MODES = {'INPUT': 0,
                     'OUTPUT': 1,
                     'ALT5': 2,
                     'ALT4': 3,
                     'ALT0': 4,
                     'ALT1': 5,
                     'ALT2': 6,
                     'ALT3': 7}

    def __init__(self, pin, gpio=None):
        """ Inherits attributes and methods from IODeviceBase.

        Args:
            pin (int): The number of the pin to use
            gpio (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        super().__init__(gpio)
        self.pin = pin

    @property
    @gpio_command
    def mode(self) -> str:
        """ The currently active pigpio mode of the pin."""
        return {value: key for key, value in self._PIGPIO_MODES.items()}[self.gpio.get_mode(self.pin)]

    @mode.setter
    @gpio_command
    def mode(self, mode):
        """ Performs validation on requested mode, then sets the mode. Raises runtime error if something goes wrong.

        Args:
            mode (str): A mode in _PIGPIO_MODES
        """
        if mode not in self._PIGPIO_MODES:
            raise ValueError("Pin mode must be one of: {}".format([*self._PIGPIO_MODES.keys()]))
        result = self.gpio.set_mode(self.pin, self._PIGPIO_MODES[mode])
        if result != 0:
            raise RuntimeError('Failed to write mode {} on pin {}'.format(mode, self.pin))

    def toggle(self):
        """ If pin is on, turn it off. If it's off, turn it on. Do not raise a warning when pin is read in this way."""
        self.write(not self.read())

    @gpio_command
    def read(self) -> int:
        """ Returns the value of the pin: usually 0 or 1 but can be overridden by subclass."""
        return self.gpio.read(self.pin)

    @gpio_command
    def write(self, value):
        """ Sets the value of the Pin. Usually 0 or 1 but behavior differs for some subclasses.

        Args:
            value: The value to write to the pin. Can be either `1` to turn on the pin or `0` to turn it off.
        """
        if value not in (0, 1):
            raise ValueError('Cannot write a value other than 0 or 1 to a Pin')
        self.gpio.write(self.pin, value)


class PWMOutput(Pin):
    """ A pin configured to output a PWM signal. Can be configured to use either a hardware-generated or
    software-generated signal. Overrides parent methods read() and write().
    """
    _DEFAULT_FREQUENCY = 20000
    _DEFAULT_SOFT_FREQ = 2000
    _HARDWARE_PWM_PINS = (12, 13, 18, 19)

    def __init__(self, pin, initial_duty=0, frequency=None, gpio=None):
        """ Inherits attributes from parent Pin, then sets PWM frequency & initial duty (use defaults if None)

        Args:
            pin (int): The number of the pin to use. Hardware PWM pins are 12, 13, 18, and 19.
            initial_duty (float): The initial duty cycle of the pin. Must be between 0 and 1.
            frequency (float): The PWM frequency to use.
            pig (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        super().__init__(pin=pin, gpio=gpio)
        self._hardware_enabled = True if self.pin in self._HARDWARE_PWM_PINS else False
        default_f = self._DEFAULT_FREQUENCY if self._hardware_enabled else self._DEFAULT_SOFT_FREQ
        self.__pwm(default_f if frequency is None else frequency, initial_duty)

    @property
    def hardware_enabled(self):
        """ Return true if this is a hardware-enabled PWM pin; False if not. The Raspberry Pi only supports hardware-
        generated PWM on pins 12, 13, 18, and 19, so generally `hardware_enabled` will be true if this is one of those,
        and false if it is not. However, `hardware_enabled` can also by dynamically set to False if for some reason
        pigpio is unable to start a hardware PWM (i.e. if the clock is unavailable or in use or something)
        """
        return self._hardware_enabled

    @property
    @gpio_command
    def frequency(self) -> float:
        """ Return the current PWM frequency active on the pin."""
        return self.gpio.get_PWM_frequency(self.pin)

    @frequency.setter
    def frequency(self, new_frequency):
        """ TODO: extend exception handling/logging to this
        Note: pigpio.pi.hardware_PWM() returns 0 if OK and an error code otherwise.
        - Tries to write hardware PWM if hardware_enabled
        - If that fails, or if not hardware_enabled, tries to write software PWM instead.

        Args:
            new_frequency (float): A new PWM frequency to use.
            """
        self.__pwm(new_frequency, self._duty())

    @property
    @gpio_command
    def duty(self) -> float:
        """ Returns the PWM duty cycle (pulled straight from pigpiod) mapped to the range [0-1] """
        return self.gpio.get_PWM_dutycycle(self.pin) / self.gpio.get_PWM_range(self.pin)

    @gpio_command
    def _duty(self) -> int:
        """ Returns the pigpio integer representation of the duty cycle."""
        return self.gpio.get_PWM_dutycycle(self.pin)

    @duty.setter
    @gpio_command
    def duty(self, duty_cycle):
        """ Sets the duty cycle.
        Args:
            duty_cycle (float): The PWM duty cycle to set. Must be between 0 and 1 (verified upon calling __pwm()).
        """
        self.__pwm(self.frequency, duty_cycle)

    def read(self) -> float:
        """ Overridden to return duty cycle instead of reading the value on the pin."""
        return self.duty

    def write(self, value):
        """ Overridden to write duty cycle.

        Args:
            value (float): See `PWMOutput.duty`
        """
        self.duty = value

    def __pwm(self, frequency, duty):
        """ Sets a PWM frequency and duty using either hardware or software generated PWM according to the value of
        `self.hardware_enabled`. If hardware_enabled, starts a hardware pwm with the requested duty. If not
        hardware_enabled, or if there is a problem setting a hardware generated PWM, starts a software PWM.

        Args:
            frequency (float): A new PWM frequency to use.
            duty (float): The PWM duty cycle to set. Must be between 0 and 1.
        """
        if not 0 <= duty <= 1:
            raise ValueError('Duty cycle must be between 0 and 1 but got {}'.format(duty))
        _duty = int(duty * self.gpio.get_PWM_range(self.pin))
        if self._hardware_enabled:
            self.__hardware_pwm(frequency, _duty)
        if not self._hardware_enabled:
            self.__software_pwm(frequency, _duty)

    @gpio_command
    def __hardware_pwm(self, frequency, duty):
        """ Used for pins where hardware pwm is available.
        -Tries to write a hardware pwm. result == 0 if it succeeds.
        -Sets hardware_enabled flag to indicate success or failure

        Args:
            frequency: A new PWM frequency to use.
            duty (int): The PWM duty cycle to set. Must be between 0 and 1.
        """
        try:
            self.gpio.hardware_PWM(self.pin, frequency, duty)
            self._hardware_enabled = True
        except Exception as e:
            self._hardware_enabled = False
            self.__software_pwm(frequency, duty)

    @gpio_command
    def __software_pwm(self, frequency, duty):
        """ Used for pins where hardware PWM is NOT available.

        Args:
            frequency: A new PWM frequency to use.
            duty (int): A pigpio integer representation of duty cycle
        """
        self.gpio.set_PWM_dutycycle(self.pin, duty)
        realized_frequency = self.gpio.set_PWM_frequency(self.pin, frequency)
        if frequency != realized_frequency:
            raise RuntimeWarning(
                'A PWM frequency of {} was requested but the best that could be done was {}'.format(
                    frequency,
                    realized_frequency
                )
            )
        self._hardware_enabled = False
