# Netgenerate

# netgenerate --rand --rand.iterations 75 --rand.connectivity 0.75 --rand.grid --rand.min-angle 90 --rand.min-distance 200 --rand.max-distance 500 --tls.guess True --default.lanenumber 2 --fringe.guess --output-file random_grid.net.xml

# Trip Generate

# python ~/sumo/tools/randomTrips.py --begin 0 --end 3600 --period 0.75 --binomial 10 --min-distance 800 --random-depart --fringe-factor 100 --validate --remove-loops --net-file random_grid.net.xml -o random_grid.trips.xml
