#!/usr/bin/env python3

import sys
print(f"Python path: {sys.path}")

try:
    from deluca.filters.spectral import SpectralFilter
    print("SpectralFilter imported successfully")
    
    # Try to create an instance
    filter = SpectralFilter(obs_dim=3, action_dim=2, num_filters=24, spectral_history_length=100)
    print("SpectralFilter created successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 