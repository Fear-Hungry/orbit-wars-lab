# Literature-Backed Heuristic Notes

This repo's current submission heuristic follows three principles reinforced by the RTS / Planet Wars literature:

1. Abstract the decision into a small number of strategic choices instead of raw micro search.
   Source: Michael Buro, "Real-Time Strategy Games: A New AI Research Challenge" (IJCAI 2003).
   Link: https://www.cs.drexel.edu/~santi/teaching/2012/CS680/papers/W2R1.pdf

2. Use local spatial scoring to value expansion, pressure, and target accessibility.
   Source: Alberto Uriarte and Santiago Ontañón, "Kiting in RTS Games Using Influence Maps" (AIIDE 2012).
   Link: https://ocs.aaai.org/ocs/index.php/AIIDE/AIIDE12/rt/printerFriendly/5497/0

3. Prefer fast, parameterized, reactive policies for Planet Wars-like domains when evaluating many matches.
   Source: Simon M. Lucas et al., "Game AI Research with Fast Planet Wars Variants" (2018).
   Link: https://arxiv.org/abs/1806.08544

The chosen submission keeps exactly that shape:

- lightweight state aggregation in `encode`
- mode selection in `policy_forward` (`expand`, `pressure`, `ffa`)
- multi-source target scoring in `decode`
- target prediction for rotating planets and sun-safe routing

An influence-field variant was tested locally during this session, but it regressed against the existing strong baseline on a short benchmark versus `weak_random`. The repo therefore keeps the stronger validated heuristic instead of shipping an unverified "improvement".
