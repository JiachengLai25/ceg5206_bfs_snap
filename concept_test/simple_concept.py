import networkx as nx

def calculate_social_distance(file_path, user1_id, user2_id, sep='\t'):
    """
    Loads SNAP ego network data (edge list format) and calculates the social distance 
    between two users (nodes).

    Args:
        file_path (str): The path to the SNAP dataset file (edge list format).
        user1_id (int/str): The ID of the first user.
        user2_id (int/str): The ID of the second user.
        sep (str): The delimiter used between node IDs in the file.

    Returns:
        int/float: The length of the shortest path (social distance) between the two users,
                   or None if the users are unreachable or do not exist.
    """
    try:
        # 1. Create an undirected graph
        G = nx.Graph()

        # 2. Load data from the edge list file
        # Note: Node IDs are often integers in SNAP datasets, but are read as strings from the file.
        # If your IDs contain non-numeric characters, they should remain as strings.
        # Assuming IDs are integers, we attempt to convert them to int type.
        try:
            G = nx.read_edgelist(file_path, 
                                 create_using=nx.Graph(), 
                                 nodetype=int, 
                                 data=(('weight', float),), 
                                 delimiter=sep)
        except ValueError:
            # If conversion fails (e.g., IDs are strings), try keeping them as strings
            print("Notice: Node IDs appear not to be purely numeric. Keeping them as string type.")
            G = nx.read_edgelist(file_path, 
                                 create_using=nx.Graph(), 
                                 nodetype=str, 
                                 data=(('weight', float),), 
                                 delimiter=sep)
            # Ensure the input user IDs match the type of the loaded graph nodes
            user1_id = str(user1_id)
            user2_id = str(user2_id)


        print(f"Network loaded successfully. Total Nodes: {G.number_of_nodes()}, Total Edges: {G.number_of_edges()}")

        # 3. Check if users exist in the network
        if user1_id not in G:
            print(f"Error: User ID {user1_id} does not exist in the network.")
            return None
        if user2_id not in G:
            print(f"Error: User ID {user2_id} does not exist in the network.")
            return None

        # 4. Calculate the shortest path length (social distance)
        try:
            # shortest_path_length uses Breadth-First Search (BFS) 
            # to find the shortest path in an unweighted graph (minimum number of edges).
            distance = nx.shortest_path_length(G, source=user1_id, target=user2_id)
            return distance
        except nx.NetworkXNoPath:
            # If the two users are in different connected components, they are unreachable
            print(f"Users {user1_id} and {user2_id} are unreachable in the network (social distance is infinite).")
            return float('inf')

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred during loading or calculation: {e}")
        return None

# --- Example Usage ---

# Replace with the path to your SNAP dataset file
DATA_FILE = 'facebook_combined.txt' 

# Assuming your data is in edge list format, e.g.:
# 0 1
# 0 2
# 1 3
# 2 4 

# Assuming you want to calculate the distance between users with ID 0 and 4
USER_A = 0
USER_B = 4

# If your node IDs are numeric, set USER_A and USER_B as integers.
# If your node IDs are strings (e.g., 'u123'), set them as strings.

social_distance = calculate_social_distance(DATA_FILE, USER_A, USER_B, sep=' ')

if social_distance is not None and social_distance != float('inf'):
    print(f"\nThe social distance between user {USER_A} and user {USER_B} is: {social_distance}")

# Example: Assuming 0 -> 2 -> 4 (distance is 2)
# ------------------
# SNAP Dataset Format Note:
# - Most ego-nets are in edge list format (source_id target_id), typically space or tab-separated.
# - For files like 'facebook_combined.txt', it is already in edge list format.
# ------------------