#!/bin/bash

for MEAN_VEHICLES_PER_HOUR in 500 600 700; do
    ARTERIAL_INTERVAL=$((3600 / MEAN_VEHICLES_PER_HOUR))
    SIDE_INTERVAL=$((ARTERIAL_INTERVAL * 3 / 5))

    NROWS=1
    NCOLS=4

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
        --interval $ARTERIAL_INTERVAL \
        --vehHeadwayTime 2

    # Replace "time": 30, with "time": 20, in the roadnetFile
    sed -i 's/"time": 30,/"time": 20,/g' "$ROADNET_FILE"

    # Process the JSON file in-place
    jq --arg interval "$SIDE_INTERVAL" '
    .vehicle[] |=
        if .route | length == 2 then
        .interval = $interval | tonumber
        else
        .
        end
    ' "$ROADNET_FILE" > tmp.json && mv tmp.json "$ROADNET_FILE"

done