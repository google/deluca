# deluca-plugin

Welcome to your new `Deluca Plugin (TM)`!

## IMPORTANT!
`deluca` uses [Python namespace
packages](https://packaging.python.org/guides/packaging-namespace-packages/) to
handle its plugins. In order to maintain this, we recommend you do not create
the file `[PLUGIN_ROOT]/deluca/__init__.py` or write code outside of
`[PLUGIN_ROOT]/deluca/plugin`.

## Getting started
1. Optional, but good to fill out `[PLUGIN_ROOT]/setup.cfg` as appropriate.
2. Start writing your code in `[PLUGIN_ROOT]/deluca/plugin`.
3. Create a pull request [here](https://github.com/google/deluca/pulls) if you'd
   like us to review your plugin and add it to the official list.