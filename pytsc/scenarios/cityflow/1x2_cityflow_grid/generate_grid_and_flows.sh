#!/bin/bash

for MEAN_VEHICLES_PER_HOUR in 500 600 700; do
    INTERVAL=$((3600 / MEAN_VEHICLES_PER_HOUR))

    NROWS=1
    NCOLS=2

    ROADNET_FILE="${NROWS}x${NCOLS}_roadnet.json"

    python /home/rohitbokade/CityFlow/tools/generator/generate_grid_scenario.py $NROWS $NCOLS \
        --rowDistance=300 \
        --columnDistance=300 \
        --intersectionWidth=20 \
        --numLeftLanes=0 \
        --numRightLanes=0 \
        --numStraightLanes=1 \
        --laneMaxSpeed=10 \
        --vehMinGap=2.5 \
        --vehMaxSpeed=10 \
        --tlPlan \
        --flowFile "${NROWS}x${NCOLS}_roadnet_${MEAN_VEHICLES_PER_HOUR}.json" \
        --roadnetFile "$ROADNET_FILE" \
        --interval $INTERVAL \
        --vehHeadwayTime 2

    # Replace "time": 30, with "time": 20, in the roadnetFile
    sed -i 's/"time": 30,/"time": 20,/g' "$ROADNET_FILE"

done