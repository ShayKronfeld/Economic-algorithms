from typing import List, Tuple

def egalitarian_allocation(valuations: List[List[int]]) -> Tuple[List[List[int]], int]:
    num_players = len(valuations)                 # Number of players (agents)
    num_items = len(valuations[0])                # Number of items to allocate

    best_result = {
        "min_value": float('-inf'),               # Best minimum utility found so far (to be maximized)
        "allocation": [[] for _ in range(num_players)],  # Initialize an empty allocation list: one empty list per player to track which items are assigned to each
    }
    best_min_player = None                        # Index of the most disadvantaged player in best result

    visited_states = set()  # ← Set for pruning identical states (Rule A)

    def backtrack(item_index: int, current_allocation: List[List[int]], player_scores: List[int]):
        """
        Recursive backtracking function to explore all possible allocations.

        Args:
            item_index: the index of the current item being considered for allocation
            current_allocation: partial allocation so far; a list of lists indicating which items each player has received
            player_scores: the total utility value each player has accumulated from their assigned items so far
        """
        nonlocal best_min_player  # Allows the recursive inner function to change the variable in the main function

        # ---------- Rule A: Prune duplicate states ----------
        # Skip this state if it was already visited (same item index + same utility vector)
        state = (item_index, tuple(player_scores))
        if state in visited_states:
            return  # identical state — skip it
        visited_states.add(state)
        # ----------------------------------------------------

        # Base case of recursion: all items have been allocated
        if item_index == num_items:
            # Compute the minimum utility among all players (i.e., the most disadvantaged player) and get their index
            min_value = min(player_scores)
            min_player = player_scores.index(min_value)

            # Update the best result if the current allocation is better (higher minimum value, or same minimum value with a smaller index of the weakest player)
            if (min_value > best_result["min_value"] or
                (min_value == best_result["min_value"] and
                 (best_min_player is None or min_player < best_min_player))):
                best_result["min_value"] = min_value

                # Store a deep copy of the current allocation as the best found so far.
                # This avoids future modifications (due to backtracking) from affecting the saved result.
                best_result["allocation"] = [list(p) for p in current_allocation]

                best_min_player = min_player
            return

        # Rule B: Optimistic Bound (Pruning based on the weakest player's potential)
        # Identify the weakest player and calculate how much value they could still get at best
        weakest_player = player_scores.index(min(player_scores))
        remaining_value = sum(valuations[weakest_player][i] for i in range(item_index, num_items))
        optimistic = player_scores[weakest_player] + remaining_value

        # If the best possible value for the weakest player in the current path is less than the current best minimum value, prune this branch
        if optimistic < best_result["min_value"]:
            return

        # Try assigning the current item to each player
        for p in range(num_players):
            current_allocation[p].append(item_index)              # Assign item to player p
            player_scores[p] += valuations[p][item_index]         # Update the player's utility

            # Recursively explore the next item
            backtrack(item_index + 1, current_allocation, player_scores)

            # Backtrack: undo the assignment to explore other possibilities
            current_allocation[p].pop()
            player_scores[p] -= valuations[p][item_index]

    # Start the backtracking process from the first item with an empty allocation and zero scores
    backtrack(0, [[] for _ in range(num_players)], [0] * num_players)
    return best_result["allocation"], best_result["min_value"]

valuations = [
    [4, 5, 6, 7, 8],
    [8, 7, 6, 5, 4]
]

allocation, min_value = egalitarian_allocation(valuations)

for i, items in enumerate(allocation):
    total = sum(valuations[i][j] for j in items)
    print(f"Player {i} gets items {items} with value {total}")
