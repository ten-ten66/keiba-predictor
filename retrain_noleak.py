#!/usr/bin/env python3
"""Retrain model without data leakage"""
import sys
sys.path.insert(0, ".")
from auto_learn import retrain_model
version = retrain_model(force=True)
print(f"RESULT: {version}")
