""" Base classes & functions used throughout vent.io.devices
"""
from collections import OrderedDict
import functools
import traceback

import time
import pigpio


def gpio_command(func):
    @functools.wraps(func)
    def exception_catcher(self, *args, **kwargs):
        result = None
        try:
            result = func(self, *args, **kwargs)
        except Exception as e:
            print(e, traceback.TracebackException.from_exception(e))
        return result

    return exception_catcher


class PigpioConnection(pigpio.pi):
    """ Subclass that extends pigpio.pi to throw an exception if there are issues connecting to the pigpio daemon."""

    def __init__(self, *args, **kwargs):
        """ Calls superclass init and checks if a connection was established; throws a RuntimeError if not.

        Args:
            *args: parameters to pass through like: pigpio.pi().__init__(*args)
            **kwargs: parameters to pass through like: pigpio.pi().__init__(**kwargs)
        """
        super().__init__(*args, **kwargs)
        if not self.connected:
            raise RuntimeError('Could not establish connection with pigpio daemon')


class IODeviceBase:
    """ Abstract base Class for pigpio handles (or whatever other GPIO library
    we end up using)

    Note: pigpio commands return -144 if an error is encountered while
    attempting to communicate with the demon. TODO would be to recognize
    when that occurs and handle it gracefully, i.e. kill the daemon,
    restart it, and reopen the python interface(s)
    """

    def __init__(self, gpio: PigpioConnection = None):
        """ Initializes the pigpio python bindings object if necessary,
        and checks that it is actually running.

        Args:
            gpio (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        self._gpio = gpio if gpio is not None else PigpioConnection(show_errors=False)
        self._handle = -1

    @property
    def gpio(self) -> PigpioConnection:
        """ The pigpio python bindings object"""
        return self._gpio

    @property
    def handle(self) -> int:
        """ Pigpiod handle associated with device (only for i2c/spi)"""
        return self._handle

    @property
    def pigpiod_ok(self) -> bool:
        """ Returns True if pigpiod is running and False if not"""
        return self.gpio.connected

    def _close(self):
        """ Closes an I2C/SPI (or potentially Serial) connection"""
        if not self.gpiopiod_ok or self.handle <= 0:
            return


class I2CDevice(IODeviceBase):
    """ A class wrapper for pigpio I2C handles. Defines several methods
    used for reading from and writing to device registers. Defines
    helper classes Register and ValueField for handling the
    manipulation of arbitrary registers.

    Note: The Raspberry Pi uses LE byte-ordering, while the outside
    world tends to use BE (at least, the sensors in use so far all do).
    Thus, bytes need to be swapped from native (LE) ordering to BE
    prior to being written to an i2c device, and bytes recieved need to
    be swapped from BE into native (LE). All methods except read_device
    and write_device perform this automatically. The methods read_device
    and write_device do NOT byteswap and return bytearrays rather than
    the unsigned 16-bit int used by the other read/write methods.
    """

    def __init__(self, i2c_address, i2c_bus, pig=None):
        """ Initializes pigpio bindings and opens i2c connection.

        Args:
            i2c_address (int): I2C address of the device. (e.g., `i2c_address=0x50`)
            i2c_bus (int): The I2C bus to use. Should probably be set to 1 on Raspberry Pi.
            pig (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        super().__init__(pig)
        self._i2c_bus = i2c_bus
        self._open(i2c_bus, i2c_address)

    @gpio_command
    def _open(self, i2c_bus, i2c_address):
        """ Opens i2c connection given i2c bus and address."""
        self._handle = self.gpio.i2c_open(i2c_bus, i2c_address)

    @gpio_command
    def _close(self):
        """ Extends superclass method. Checks that pigpiod is connected
        and if a handle has been set - if so, closes an i2c connection.
        """
        super()._close()
        self.gpio.i2c_close(self.handle)

    @gpio_command
    def read_device(self, count=2) -> tuple:
        """ Read a specified number of bytes directly from the the device without specifying or changing the register.
        Does NOT perform LE/BE conversion.

        Args:
            count (int): The number of bytes to read from the device.

        Returns:
            tuple: a tuple of the number of bytes read and a bytearray containing the bytes. If there was an error the
            number of bytes read will be less than zero (and will contain the error code).
        """
        return self.gpio.i2c_read_device(self.handle, count)

    @gpio_command
    def write_device(self, word, signed=False):
        """ Write 2 bytes to the device without specifying register. DOES perform LE/BE conversion.

        Args:
            word (int): The integer representation of the data to write.
            signed (bool): Whether or not `word` is signed.
        """
        self.gpio.i2c_write_device(
            self.handle,
            native16_to_be(word, signed=signed)
        )

    @gpio_command
    def read_register(self, register, signed=False) -> int:
        """ Read 2 bytes from the specified register and byteswap the result.

        Args:
            register (int): The index of the register to read.
            signed (bool): Whether or not the data to read is expected to be signed.

        Returns:
            int: integer representation of 16 bit register contents.
        """
        return be16_to_native(self.gpio.i2c_read_i2c_block_data(
            self.handle,
            register,
            count=2
        ), signed=signed)

    @gpio_command
    def write_register(self, register, word, signed=False):
        """ Write 2 bytes to the specified register. Byteswaps.

        Args:
            register (int): The index of the register to write to
            word (int): The unsigned 16 bit integer to write to the register (must be consistent with 'signed')
            signed (bool): Whether or not 'word' is signed
        """
        self.gpio.i2c_write_i2c_block_data(
            self.handle,
            register,
            native16_to_be(word, signed=signed)
        )

    class Register:
        """ Describes a writable configuration register. Has dynamically
        defined attributes corresponding to the fields described by the
        passed arguments. Takes as arguments two tuples of equal length,
        the first of which names each field and the second being a tuple
        of tuples containing the (human readable) possible settings &
        values for each field.

        Note: The initializer reverses the fields & their values because
        a human reads the register, as drawn in the datasheet, from left
        to right - however, the fields furthest to the left are the most
        significant bits of the register.
        """

        def __init__(self, fields, values):
            """ Initializer which loads (dynamically defined) attributes from tuples.

            Args:
                fields (tuple): A tuple containing the names of the register's value fields
                values (tuple): A tuple of tuples containing the possible values for each value field. Length must match
                    the length of fields. If there are redundant values for a field specified in the datasheet, be sure
                    to include them. (e.g., a field takes values `A: 0b00`, `B: 0b01`, and `C: 0b10`; but the value for
                    `0b11` is either not specified by the datasheet or is listed redundantly as `C: 0b11` -> `values`
                    should list both the 3rd and 4th possible values as 'C' like so: ('A', 'B', 'C', 'C')
            """
            if len(fields) != len(values):
                raise ValueError('fields and values must contain the same number of elements')
            self.fields = fields
            offset = 0
            for fld, val in zip(reversed(fields), reversed(values)):
                setattr(
                    self,
                    fld,
                    self.ValueField(
                        offset,
                        len(val) - 1,
                        OrderedDict(zip(val, range(len(val))))
                    )
                )
                offset += (len(val) - 1).bit_length()

        def unpack(self, cfg) -> OrderedDict:
            """ Given the contents of a register in integer form, returns a dict of fields and their current settings.

            Args:
                cfg (int): An integer representing a possible configuration value for the register
            """
            return OrderedDict(zip(
                self.fields,
                (getattr(
                    getattr(self, field),
                    'unpack')(cfg) for field in self.fields)
            ))

        def pack(self, cfg, **kwargs) -> int:
            """ Given an initial integer representation of a register and an arbitrary number of field=value settings,
            returns an integer representation of the register incorporating the new settings.

            Args:
                cfg (int): An integer representing a possible configuration value for the register
                **kwargs: The register fields & values to patch into cfg. Takes keyword arguments of the form:
                    `field=value`
            """
            for field, value in kwargs.items():
                if hasattr(self, field) and value is not None:
                    cfg = getattr(getattr(self, field), 'insert')(cfg, value)
            return cfg

        class ValueField:
            """ Describes a configurable value field in a writable register."""

            def __init__(self, offset, mask, values):
                """ Instantiates a value field of a register given the bit offset, mask, and list of possible values.

                Args:
                    offset (int): The offset bits of the value field in the register, i.e. the distance from LSB
                    mask (int): integer representation of the value field mask (w/o offset)
                    values (OrderedDict): The possible values that the field can take.
                """
                self._offset = offset
                self._mask = mask
                self._values = values
                self._reversed_values = OrderedDict(map(reversed, self._values.items()))

            def unpack(self, cfg):
                """ Extracts the ValueField's setting from cfg & returns the result in a human readable form.

                Args:
                    cfg (int): An integer representing a possible configuration value for the register
                """
                return self._reversed_values[self.extract(cfg)]

            def extract(self, cfg) -> int:
                """ Extracts setting from passed 16-bit config & returns integer representation.

                Args:
                    cfg (int): An integer representing a possible configuration value for the register
                """
                return (cfg & (self._mask << self._offset)) >> self._offset

            def pack(self, value) -> int:
                """ Takes a human-readable ValueField setting and returns the corresponding bit-shifted integer.

                Args:
                    value (int): The integer representation of `value` bit-shifted by the ValueField's offset

                Returns:
                    int: The integer representation of the ValueField setting according to `value`
                """
                if value not in self._values.keys():
                    raise ValueError("ValueField must be one of: {}".format(self._values.keys()))
                return self._values[value] << self._offset

            def insert(self, cfg, value) -> int:
                """ Validates and performs bitwise replacement with the human-readable ValueField setting and integer
                representation of the register configuration.

                Args:
                    cfg (int): An integer representing a possible configuration value for the register
                    value (object): The human readable representation of the desired ValueField setting. Must match a
                        value in ValueField._values; if not, throws a ValueError

                Returns:
                    int: The integer representation of the Register's configuration with the value of ValueField patched
                        according the `value`
                """
                if value not in self._values.keys():
                    raise ValueError("ValueField must be one of: {}".format(self._values.keys()))
                return (cfg & ~(self._mask << self._offset)) | (self._values[value] << self._offset)


class ADS1015(I2CDevice):
    """ ADS1015 12 bit, 4 Channel Analog to Digital Converter.
    Datasheet:
        http://www.ti.com/lit/ds/symlink/ads1015.pdf?&ts=1589228228921

    Default Values:
     Default configuration for vent:     0xC3E3
     Default configuration on power-up:  0x8583
    """
    _DEFAULT_ADDRESS = 0x48
    _DEFAULT_VALUES = {'MUX': 0, 'PGA': 4.096, 'MODE': 'SINGLE', 'DR': 3300}
    _TIMEOUT = 1
    """ Address Pointer Register (write-only) """
    _POINTER_FIELDS = ('P',)
    _POINTER_VALUES = (
        (
            'CONVERSION',
            'CONFIG',
            'LO_THRESH',
            'HIGH_THRESH'
        ),
    )

    """ Config Register (R/W) """
    _CONFIG_FIELDS = (
        'OS',
        'MUX',
        'PGA',
        'MODE',
        'DR',
        'COMP_MODE',
        'COMP_POL',
        'COMP_LAT',
        'COMP_QUE'
    )
    _CONFIG_VALUES = (
        ('NO_EFFECT', 'START_CONVERSION'),
        ((0, 1), (0, 3), (1, 3), (2, 3), 0, 1, 2, 3),
        (6.144, 4.096, 2.048, 1.024, 0.512, 0.256, 0.256, 0.256),
        ('CONTINUOUS', 'SINGLE'),
        (128, 250, 490, 920, 1600, 2400, 3300, 3300),
        ('TRADIONAL', 'WINDOW'),
        ('ACTIVE_LOW', 'ACTIVE_HIGH'),
        ('NONLATCHING', 'LATCHING'),
        (1, 2, 3, 'DISABLE')
    )
    USER_CONFIGURABLE_FIELDS = ('MUX', 'PGA', 'MODE', 'DR')
    """ Note:
    The Conversion Register is read-only and contains a 16bit
    representation of the requested value (provided the conversion is
    ready).

    The Lo-thresh & Hi-thresh Registers are not Utilized here. However,
    their function and usage are described in the datasheet. Should you
    want to extend the functionality implemented here.
    """

    def __init__(self, address=_DEFAULT_ADDRESS, i2c_bus=1, gpio=None):
        """ Initializes registers: Pointer register is write only,
        config is R/W. Sets initial value of _last_cfg to what is
        actually on the ADS.Packs default settings into _cfg, but does
        not actually write to ADC - that occurs when read_conversion()
        is called.

        Args:
            address (int): I2C address of the device. (e.g., `i2c_address=0x48`)
            i2c_bus (int): The I2C bus to use. Should probably be set to 1 on Raspberry Pi.
            gpio (PigpioConnection): pigpiod connection to use; if not specified, a new one is established
        """
        super().__init__(address, i2c_bus, gpio)
        self.pointer = self.Register(self._POINTER_FIELDS, self._POINTER_VALUES)
        self._config = self.Register(self._CONFIG_FIELDS, self._CONFIG_VALUES)
        self._last_cfg = self._read_last_cfg()
        self._cfg = self._config.pack(cfg=self._last_cfg, **self._DEFAULT_VALUES)

    def read_conversion(self, **kwargs) -> float:
        """ Returns a voltage (expressed as a float) corresponding to a channel on the ADC.
        The channel to read from, along with the gain, mode, and sample rate of the conversion may be may be  specified
        as optional parameters. If read_conversion() is called with no parameters, the resulting voltage corresponds to
        the channel last read from and the same conversion settings.

        Args:
            MUX: The pin to read from in single channel mode: e.g., `0, 1, 2, 3`
                or, a tuple of pins over which to make a differential reading.
                e.g., `(0, 1), (0, 3), (1, 3), (2, 3)`
            PGA: The full scale voltage (FSV) corresponding to a programmable gain setting.
                e.g., `(6.144, 4.096, 2.048, 1.024, 0.512, 0.256, 0.256, 0.256)`
            MODE: Whether to set the ADC to continuous conversion mode, or operate in single-shot mode.
                e.g., `'CONTINUOUS', 'SINGLE'`
            DR: The data rate to make the conversion at; units: samples per second.
                e.g., `8, 16, 32, 64, 128, 250, 475, 860`
        """
        return (
                self._read_conversion(**kwargs)
                * self.config.PGA.unpack(self.cfg) / 32767
        )

    def print_config(self) -> OrderedDict:
        """ Returns the human-readable configuration for the next read.

        Returns:
            OrderedDict: an ordered dictionary of the form {field: value}, ordered from MSB -> LSB
        """
        return self.config.unpack(self.cfg)

    @property
    def config(self):
        """ Returns the Register object of the config register.

        Returns:
            vent.io.devices.I2CDevice.Register: The Register object initialized for the ADS1115.
        """
        return self._config

    @property
    def cfg(self) -> int:
        """ Returns the contents (as a 16-bit unsigned integer) of the configuration that will be written to the config
        register when read_conversion() is next called.
        """
        return self._cfg

    def _read_conversion(self, **kwargs) -> int:
        """ Backend for read_conversion. Returns the contents of the 16-bit conversion register as an unsigned integer.

        If no parameters are passed, one of two things can happen:

            1)  If the ADC is in single-shot (mode='SINGLE') conversion
                mode, _last_cfg is written to the config register; once
                the ADC indicates it is ready, the contents of the
                conversion register are read and the result is returned.
            2)  If the ADC is in CONTINUOUS mode, the contents of the
                conversion register are read immediately and returned.

        If any of channel, gain, mode, or data_rate are specified as
        parameters, a new _cfg is packed and written to the config
        register; once the ADC indicates it is ready, the contents of
        the conversion register are read and the result is returned.

        Note: In continuous mode, data can be read from the conversion
        register of the ADS1115 at any time and always reflects the
        most recently completed conversion. So says the datasheet.

        Args:
            **kwargs: see documentation of vent.io.devices.ADS1115.read_conversion
        """
        self._cfg = self._config.pack(cfg=self.cfg, **kwargs)
        mode = self.print_config()['MODE']
        if self._cfg != self._last_cfg or mode == 'SINGLE':
            self.write_register(self.pointer.P.pack('CONFIG'), self.cfg)
            self._last_cfg = self.cfg
            data_rate = self._config.DR.unpack(self.cfg)
            '''while not (self._ready() or mode == 'CONTINUOUS'):
                # TODO: Needs timout
                tick = time.time()
                while (time.time() - tick) < (1 / data_rate):
                    pass  # TODO: implement asyncio.sleep()'''
        return self.read_register(self.pointer.P.pack('CONVERSION'), signed=True)

    def _read_last_cfg(self) -> int:
        """ Reads the config register and returns the contents as a 16-bit unsigned integer;
        updates internal record _last_cfg.
        """
        self._last_cfg = self.read_register(self.pointer.P.pack('CONFIG'))
        return self._last_cfg

    def _ready(self) -> bool:
        """ Return status of ADC conversion; True indicates the conversion is complete and the results ready to be read.
        """
        return bool(self.read_register(self.pointer.P.pack('CONFIG')) >> 15)


def be16_to_native(data, signed=False) -> int:
    """ Unpacks a bytes-like object respecting big-endianness of outside world and returns an int according to signed.

    Args:
        data: bytes-like object. The data to be unpacked & converted
        signed (bool): Whether or not `data` is signed
    """
    return int.from_bytes(data[1][:2], 'big', signed=signed)


def native16_to_be(word, signed=False) -> bytes:
    """ Packs an int into bytes after swapping endianness.

    Args:
        signed (bool): Whether or not `data` is signed
        word (int): The integer representation to converted and packed into bytes
    """
    return word.to_bytes(2, 'big', signed=signed)
