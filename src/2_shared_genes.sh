#!/bin/bash

GLOBAL_MAD=$1
COMMON_GENES=$2

echo "Shared genes:"
comm -12 \
    <(tail -n +2 "${GLOBAL_MAD}" | sort) \
    <(tail -n +2 "${COMMON_GENES}" | sort) \
    | wc -l