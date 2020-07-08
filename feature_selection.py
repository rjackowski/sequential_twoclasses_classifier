from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils import generate_possible_subsets


def best_first_feature_selection(X, Y, estimator,
                                 max_no_improvement_iterations=5):
    features = list(range(X.shape[1]))

    open = []
    closed = []

    base_score = cross_val_score(estimator, X, Y, scoring='accuracy',
                                 cv=StratifiedKFold(n_splits=5)).mean()

    open.append((features, base_score))
    best = (features, base_score)

    tries_without_improvement = 0
    while True:
        candidate = next(
            x for x in open if x[1] == max(open, key=lambda tpl: tpl[1])[1])

        open.remove(candidate)
        closed.append(candidate)

        tries_without_improvement += 1

        if candidate[1] > best[1]:
            best = candidate
            tries_without_improvement = 0

        feature_subsets = generate_possible_subsets(candidate[0])

        for feature_subset in feature_subsets:
            if feature_subset not in [x[0] for x in open] + [x[0] for x in
                                                             closed]:
                score = cross_val_score(estimator, X[:, feature_subset], Y,
                                        scoring='accuracy',
                                        cv=StratifiedKFold(n_splits=5)).mean()

                open.append((feature_subset, score))

        if tries_without_improvement > max_no_improvement_iterations or len(
                open) == 0:
            break

    return best[0]


def greedy_cost_sensitive_feature_selection(X, Y, estimator, costs):
    features = list(range(X.shape[1]))

    base_score = cross_val_score(estimator, X, Y, scoring='accuracy',
                                 cv=StratifiedKFold(n_splits=5)).mean()

    total_cost = sum(costs)

    best = (features, total_cost)

    found_better = False

    while True:
        feature_subsets = generate_possible_subsets(best[0])

        for feature_subset in feature_subsets:
            score = cross_val_score(estimator, X[:, feature_subset], Y,
                                    scoring='accuracy',
                                    cv=StratifiedKFold(n_splits=5)).mean()

            if score >= base_score:
                cost = sum(costs[feature_subset])

                if cost < best[1]:
                    best = (feature_subset, cost)
                    found_better = True

        if not found_better:
            break

        found_better = False

    return best[0]


def best_first_cost_sensitive_feature_selection(X, Y, estimator, costs,
                                                max_no_improvement_iterations=5):
    features = list(range(X.shape[1]))

    open = []
    closed = []

    base_score = cross_val_score(estimator, X, Y, scoring='accuracy',
                                 cv=StratifiedKFold(n_splits=5)).mean()

    total_cost = sum(costs)

    open.append((features, total_cost))
    best = (features, total_cost)

    tries_without_improvement = 0

    while True:
        candidate = next(
            x for x in open if
            x[1] == min(open, key=lambda tpl: tpl[1])[1])

        open.remove(candidate)
        closed.append(candidate)

        tries_without_improvement += 1

        if candidate[1] < best[1]:
            best = candidate
            tries_without_improvement = 0

        feature_subsets = generate_possible_subsets(candidate[0])

        for feature_subset in feature_subsets:
            if feature_subset not in [x[0] for x in open] + [x[0] for x in
                                                             closed]:
                score = cross_val_score(estimator, X[:, feature_subset], Y,
                                        scoring='accuracy',
                                        cv=StratifiedKFold(n_splits=5)).mean()
                cost = sum(costs[feature_subset])

                if score >= base_score:
                    open.append((feature_subset, cost))
                else:
                    closed.append((feature_subset, cost))

        if tries_without_improvement == max_no_improvement_iterations or len(
                open) == 0:
            break

    return best[0]
