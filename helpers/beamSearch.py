import copy

# greedy decoder
def greedy_decoder(data_c, route, curr_max):
    max = float("-inf")
    j = 0
    for i in range(0, route.num_nodes):
        if max + curr_max < route.data_p[route.croute[-1]][i] + curr_max and route.check_next_node(i, data_c[i]):
            max = route.data_p[route.croute[-1]][i]
            j = i

    curr_max = curr_max + max
    route.add_node(j, data_c[j])
    if not route.is_terminal_route():
      return greedy_decoder(data_c, route, curr_max)
    else:
      return route

# beam search decoder
def beam_search_decoder(data_c, k, start_point):
    sequences = [start_point]
    return predict_next(data_c, k, sequences)

def predict_next(data_c, k, sequences):
    new_sequences = []
    new_route = []
    for route in sequences:
      if route.num_nodes > len(set(route.croute)) or (route.croute[-1] != 0 and route.num_nodes == len(set(route.croute))):
        for i in range(0, route.num_nodes):
            if route.check_next_node(i, data_c[i]):
                new_route = copy.deepcopy(route)
                new_route.add_node(i, data_c[i])
                new_sequences.append(new_route)
      else:
         new_sequences.append(route)

    ordered = sorted(set(new_sequences), key=lambda x: x.score)
    sequences = ordered[:k]
    num_nodes = len(data_c)
    last_node = 0
    for tab in sequences:
      if last_node < tab.croute[-1]:
         last_node = tab.croute[-1] 
      x = len(set(tab.croute))
      if num_nodes > x:
          num_nodes = x

    # jeśli istnieje nie zamknięta trasa, szukamy kolejnego wierzchołka      
    if num_nodes < len(data_c) or (last_node != 0 and num_nodes == len(data_c)):
      return predict_next(data_c, k, sequences)
    else:
      return sequences

def score_length(data_d, route):
    score = 0.0
    for i in range(1,len(route)):
      score = score + data_d[route[i-1]][route[i]]
    return round(score,4)

def print_all_routes(y_dist, y_c, k, start_point):
  result = beam_search_decoder(y_c, k, start_point)
  best_route = []
  max_length = score_length(y_dist,result[0].croute)
  for route in result:
    print(f'Trasa: {route.croute} Długość: {score_length(y_dist,route.croute)} Score: {route.score}')

  return best_route, max_length

def get_best_route_bs(y_dist, y_c, k, start_point, not_greedy = False):
  if k > 1 or not_greedy:
    result = beam_search_decoder(y_c, k, start_point)
    best_route = []
    best_length = float("inf")
    for route in result:
      route_length = score_length(y_dist,route.croute)
      if route_length < best_length:
        best_route = route
        best_length = route_length
  else:
     route = copy.deepcopy(start_point)
     best_route = greedy_decoder(y_c, route, 0)
     best_length = score_length(y_dist, best_route.croute)

  return best_route, best_length

def pick_best_child(y_dist, y_c, k, route):
  best_route, l = get_best_route_bs (y_dist, y_c, k, route)
  return best_route.croute[len(route.croute)]



# from cvrproute import CVRPRoute

# new = CVRPRoute(len(c), copy.deepcopy(p))

# width = 4

# while len(set(new.croute)) < len(c):
#    x = pick_best_child(d,c,width,new)
#    new.add_node(x, c[x])
#    print(x)

# print_all_routes(d,c,width,new)

# new = CVRPRoute(len(c), p)
# r, d = get_best_route_bs(d,c,width,new)
# print(r.croute)
# print(d)

