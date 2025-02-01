"""Capacited Vehicles Routing Problem (CVRP)."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math 

def distance(cord1,cord2):
    return round(math.sqrt((cord2[0]-cord1[0])**2+(cord2[1]-cord1[1])**2))

def create_data_model(distance_matrix, demands, vehicle_capacities, num_vehicles):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = distance_matrix
    data["demands"] = demands
    data["vehicle_capacities"] = vehicle_capacities
    data["num_vehicles"] = num_vehicles
    data["depot"] = 0
    return data

def load_data(size, index):
    lines = open("dataset/CVRP"+str(size)+".txt", "r").readlines()
    for line_num, line in enumerate(lines):
        if line_num == index:
            cord = []
            demands = []
            vehicle_capacities = []
            line = line.split(" ")  # Split into list
            num_vehicles = int(line[1])
            for i in range (2,num_vehicles+2):
                vehicle_capacities.append(int(line[i]))
            for i in range (num_vehicles+2,size*3+num_vehicles+2,3):
                cord.append([int(line[i]),int(line[i+1])])
                demands.append(int(line[i+2]))
            break

    return cord, demands, vehicle_capacities, num_vehicles


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total load of all routes: {total_load}")

def save_solution(data, manager, routing, solution, cord, demands, vehicle_capacities, num_vehicles):
    size = len(demands)
    f = open("dataset/CVRP"+str(size)+"_train.txt", "a")

    f.write(str(size)+" "+str(num_vehicles)+" ")
    for j in range(0,num_vehicles):
        f.write(str(vehicle_capacities[j])+" ")
    for j in range(0,size):
        for z in range(0,2):
            f.write(str(cord[j][z])+" ")
        f.write(str(demands[j])+" ")
    f.write("output: ")

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = ""
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f"{node_index} "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)} "
        f.write(f"{plan_output}")
    f.write("\n")
    f.close()

def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    stop = 0
    start = 0
    for index in range(start,20000):
        size = 49
        cord, demands, vehicle_capacities, num_vehicles = load_data(size, index)
        distance_matrix = [[distance(cord[i],cord[j]) for i in range(size)] for j in range(size)]
        data = create_data_model(distance_matrix, demands, vehicle_capacities, num_vehicles)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_capacities"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(60) #3 15 60

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            save_solution(data, manager, routing, solution, cord, demands, vehicle_capacities, num_vehicles)
            stop += 1
            if stop == 10000:
                break


if __name__ == "__main__":
    main()  
