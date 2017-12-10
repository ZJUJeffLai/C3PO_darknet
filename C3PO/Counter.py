class Counter:
    def __init__(self):
        self.pedestrian = 0
        self.bike = 0
        self.vehicle = 0

    def add_pedestrian(self):
        self.pedestrian+=1

    def add_bike(self):
        self.bike+=1

    def add_vehicle(self):
        self.vehicle+=1

    def stats(self):
        return ("pedestrian: " + str(self.pedestrian) + " vehile: " + str(self.vehicle) + " bike: " + str(self.bike))

    def stats_terminal_report(self):
        print("pedestrian: " + str(self.pedestrian) + " vehile: " + str(self.vehicle) + " bike: " + str(self.bike))