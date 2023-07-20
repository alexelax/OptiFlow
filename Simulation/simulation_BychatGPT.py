import os
import sys
import traci


laneDetectors=[
    "E0_0D",
    "E0_1D",
    "-E4_0D",
    "-E5_0D",
    "E2_0D",
    "E2_1D",
    "-E1_0D",
]
def run_simulation(simulation_duration):
    # Path to the SUMO binary
    sumo_binary = "sumo-gui"

    # Path to the SUMO configuration file (.sumocfg)
    sumo_config_file = "data/test.sumocfg"

    # Start the SUMO simulation as a subprocess
    sumo_cmd = [sumo_binary, "-c", sumo_config_file]
    sumo_process = traci.start(sumo_cmd)

    # Main simulation loop
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Replace "intersection_id" with the ID of the intersection you want to monitor
        intersection_id = "J2"

        # Get the number of vehicles for each lane at the specified intersection
        num_vehicles_per_lane = {}
        #lanes = traci.trafficlight.getControlledLanes(intersection_id)
        for lane in laneDetectors:
            num_vehicles_per_lane[lane] = traci.lanearea.getLastStepVehicleNumber(lane)
            #TODO: c'è una giunzione a destra, devo trovare il modo di prendere quei veicoli??? bha

        # TODO: Implement your logic to interact with the traffic lights based on the current situation

    # End the simulation and close the connection
    traci.close()

def main():
    # Specify the duration of the simulation in seconds
    simulation_duration = 3600  # One hour in this example

    # Run the simulation
    run_simulation(simulation_duration)

    # Get the simulation report
    total_simulation_time = traci.simulation.getTime()
    average_waiting_times = {}  # A dictionary to store average waiting time for each vehicle
    max_waiting_time = 0

    # Iterate over all vehicles
    for vehicle_id in traci.vehicle.getIDList():
        # Get the waiting time for the current vehicle
        waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)

        # Calculate the average waiting time for the current vehicle
        average_waiting_times[vehicle_id] = waiting_time / traci.vehicle.getTripTime(vehicle_id)

        # Update the maximum waiting time if needed
        max_waiting_time = max(max_waiting_time, average_waiting_times[vehicle_id])

    # Print the simulation report
    print("Simulation report:")
    print("Total simulation time:", total_simulation_time)
    print("Average waiting times for each vehicle:", average_waiting_times)
    print("Maximum waiting time:", max_waiting_time)

if __name__ == "__main__":
    main()
