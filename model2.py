"""
Heads-up no-limit hold'em bot.

Ideas:
  * Fast Monte Carlo equity (own 7-card evaluator, ~14x faster than
    card_utils.evaluate), so many more simulations fit in each move.
  * Opponents are modelled, not assumed random: per team (game_state.teams) it
    learns how often they raise, how often they fold to aggression, and how
    strong the hands they showed down after betting were. When an opponent
    raises, their range is narrowed to what that team has shown, so a tight
    bot's raise is respected and a maniac's raise is not.
  * Every move (fold / call / each raise size) gets an expected value that
    includes fold equity, the opponent's calling range and re-raise risk; the
    best one is played. Bluffs only happen against opponents who fold.
"""

import random

from player_action import DEALER, PlayerAction

SIMS = 1000          # Monte Carlo samples per equity estimate (heads-up)
DEBUG = False        # True: let exceptions escape (local testing only)

_RANKS = "23456789TJQKA"
_SUIT = {"C": 0, "D": 1, "H": 2, "S": 3}

# Equity of each starting-hand class against one random hand.
_TABLE_TEXT = "AA:0.849 KK:0.815 QQ:0.804 JJ:0.773 TT:0.752 99:0.724 88:0.694 AKs:0.678 AQs:0.662 77:0.659 AJs:0.653 AQo:0.65 AKo:0.648 ATs:0.644 AJo:0.638 KQs:0.637 A9s:0.634 66:0.633 ATo:0.627 KJs:0.624 A8s:0.619 KTs:0.617 A7s:0.61 A9o:0.609 KQo:0.608 K9s:0.605 55:0.605 A5s:0.601 KTo:0.6 KJo:0.599 QJs:0.597 A8o:0.594 QTs:0.593 A7o:0.592 A6s:0.591 K8s:0.588 A6o:0.587 A3s:0.587 Q9s:0.585 A4s:0.583 QJo:0.582 K9o:0.581 JTs:0.577 A5o:0.575 K7s:0.573 QTo:0.573 K6s:0.571 44:0.57 J9s:0.568 A2s:0.567 A3o:0.565 A4o:0.564 Q8s:0.563 K5s:0.56 JTo:0.56 K7o:0.556 K4s:0.556 Q9o:0.555 K8o:0.554 Q7s:0.551 A2o:0.546 K3s:0.544 Q8o:0.543 T9s:0.543 K6o:0.542 33:0.541 J8s:0.539 K2s:0.537 Q6s:0.535 K5o:0.534 J9o:0.53 K4o:0.527 Q5s:0.527 J7s:0.526 T8s:0.523 Q7o:0.522 J8o:0.519 Q4s:0.516 T7s:0.516 Q3s:0.515 K3o:0.513 T9o:0.511 Q2s:0.51 K2o:0.509 98s:0.509 22:0.506 Q6o:0.505 J7o:0.499 J5s:0.498 J6s:0.494 T6s:0.493 97s:0.492 T8o:0.491 Q4o:0.489 Q5o:0.488 J4s:0.488 T7o:0.484 J3s:0.482 J6o:0.481 87s:0.481 98o:0.479 Q3o:0.475 J5o:0.475 J2s:0.474 97o:0.474 Q2o:0.471 T5s:0.468 96s:0.465 86s:0.465 J4o:0.463 T3s:0.463 T4s:0.462 T6o:0.46 87o:0.458 95s:0.455 T2s:0.448 J3o:0.447 76s:0.444 85s:0.442 96o:0.441 J2o:0.436 T5o:0.436 94s:0.433 84s:0.433 75s:0.431 T4o:0.43 93s:0.429 95o:0.428 76o:0.428 86o:0.426 T3o:0.424 65s:0.423 92s:0.422 T2o:0.419 83s:0.416 74s:0.415 54s:0.414 85o:0.413 64s:0.411 94o:0.409 82s:0.406 93o:0.405 75o:0.401 65o:0.401 73s:0.397 53s:0.397 63s:0.396 84o:0.391 92o:0.389 74o:0.384 54o:0.384 43s:0.384 72s:0.383 64o:0.378 52s:0.375 42s:0.375 83o:0.371 73o:0.371 53o:0.371 62s:0.369 82o:0.368 63o:0.364 32s:0.358 43o:0.355 72o:0.351 62o:0.34 52o:0.339 42o:0.327 32o:0.32"


# --------------------------------------------------------------------------
# Hand evaluation. Cards are ints: (rank - 2) * 4 + suit, rank 2..14.
# --------------------------------------------------------------------------

def _straight_high(mask):
    for h in range(14, 4, -1):
        if (mask >> (h - 4)) & 31 == 31:
            return h
    return 0


def _score(cs):
    """Strength of the best 5-card hand among 5 to 7 cards. Higher wins,
    equal splits."""
    cnt = [0] * 15
    sc = [0, 0, 0, 0]
    for c in cs:
        cnt[(c >> 2) + 2] += 1
        sc[c & 3] += 1

    flush_suit = -1
    for s in range(4):
        if sc[s] >= 5:
            flush_suit = s

    if flush_suit >= 0:
        fr = [(c >> 2) + 2 for c in cs if (c & 3) == flush_suit]
        m = 0
        for r in fr:
            m |= 1 << r
        if 14 in fr:
            m |= 2
        sh = _straight_high(m)
        if sh:
            return (8 << 20) | (sh << 16)

    quads = []
    trips = []
    pairs = []
    singles = []
    mask = 0
    for r in range(14, 1, -1):
        n = cnt[r]
        if n:
            mask |= 1 << r
            if n == 4:
                quads.append(r)
            elif n == 3:
                trips.append(r)
            elif n == 2:
                pairs.append(r)
            else:
                singles.append(r)

    if quads:
        q = quads[0]
        rest = [r for r in range(14, 1, -1) if cnt[r] and r != q]
        return (7 << 20) | (q << 16) | (rest[0] << 12)

    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        if len(trips) > 1:
            p = max(trips[1], pairs[0]) if pairs else trips[1]
        else:
            p = pairs[0]
        return (6 << 20) | (t << 16) | (p << 12)

    if flush_suit >= 0:
        fr = sorted(((c >> 2) + 2 for c in cs if (c & 3) == flush_suit),
                    reverse=True)[:5]
        v = 5 << 20
        shift = 16
        for r in fr:
            v |= r << shift
            shift -= 4
        return v

    if cnt[14]:
        mask |= 2
    sh = _straight_high(mask)
    if sh:
        return (4 << 20) | (sh << 16)

    if trips:
        t = trips[0]
        kick = [r for r in range(14, 1, -1) if cnt[r] and r != t][:2]
        return (3 << 20) | (t << 16) | (kick[0] << 12) | (kick[1] << 8)

    if len(pairs) >= 2:
        a, b = pairs[0], pairs[1]
        kick = [r for r in range(14, 1, -1) if cnt[r] and r != a and r != b]
        return (2 << 20) | (a << 16) | (b << 12) | (kick[0] << 8)

    if pairs:
        p = pairs[0]
        kick = [r for r in range(14, 1, -1) if cnt[r] and r != p][:3]
        return ((1 << 20) | (p << 16) | (kick[0] << 12) | (kick[1] << 8)
                | (kick[2] << 4))

    v = 0
    shift = 16
    for r in singles[:5]:
        v |= r << shift
        shift -= 4
    return v


# --------------------------------------------------------------------------
# Preflop ranking: all 1326 starting combos, strongest first.
# --------------------------------------------------------------------------

def _build_ranking():
    eq = {}
    for item in _TABLE_TEXT.split():
        name, value = item.split(":")
        eq[name] = float(value)
    names = sorted(eq, key=lambda k: -eq[k])
    combos = []
    pct = {}
    for name in names:
        hi = _RANKS.index(name[0])
        lo = _RANKS.index(name[1])
        start = len(combos)
        if hi == lo:
            for s1 in range(4):
                for s2 in range(s1 + 1, 4):
                    combos.append((hi * 4 + s1, lo * 4 + s2))
        elif name[2] == "s":
            for s in range(4):
                combos.append((hi * 4 + s, lo * 4 + s))
        else:
            for s1 in range(4):
                for s2 in range(4):
                    if s1 != s2:
                        combos.append((hi * 4 + s1, lo * 4 + s2))
        # Where this class sits in the ranking: 0 = best, 1 = worst.
        pct[name] = (start + len(combos)) / 2.0 / 1326.0
    return combos, pct


_COMBOS, _CLASS_PCT = _build_ranking()


def _class_pct(c1, c2):
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    name = _RANKS[r1] + _RANKS[r2]
    if r1 != r2:
        name += "s" if (c1 & 3) == (c2 & 3) else "o"
    return _CLASS_PCT[name]


def _card_int(card):
    return (card.value - 2) * 4 + _SUIT[card.suit]


# --------------------------------------------------------------------------
# Monte Carlo equity against opponent ranges.
#   spec (0, None)    random hand
#   spec (1, pool)    hand drawn from a list of preflop combos
#   spec (2, cutoff)  random hand whose made strength on the board >= cutoff
# --------------------------------------------------------------------------

def _two(rand, used):
    while True:
        a = rand(52)
        if a not in used:
            break
    while True:
        b = rand(52)
        if b not in used and b != a:
            break
    return a, b


def _cutoff(hero, board, q, rng, n=160):
    """Made-hand strength that the top fraction q of hands reaches."""
    rand = rng.randrange
    used = set(hero)
    used.update(board)
    scores = []
    for _ in range(n):
        a, b = _two(rand, used)
        scores.append(_score(board + [a, b]))
    scores.sort()
    idx = int((1.0 - q) * n)
    return scores[min(n - 1, max(0, idx))]


def _equity(hero, board, specs, sims, rng):
    rand = rng.randrange
    base = set(hero)
    base.update(board)
    missing = 5 - len(board)
    total = 0.0
    for _ in range(sims):
        used = set(base)
        hands = []
        for mode, data in specs:
            if mode == 1:
                ok = False
                n = len(data)
                for _t in range(40):
                    a, b = data[rand(n)]
                    if a not in used and b not in used:
                        ok = True
                        break
                if not ok:
                    a, b = _two(rand, used)
            elif mode == 2:
                for _t in range(30):
                    a, b = _two(rand, used)
                    if _score(board + [a, b]) >= data:
                        break
            else:
                a, b = _two(rand, used)
            used.add(a)
            used.add(b)
            hands.append((a, b))
        full = list(board)
        for _m in range(missing):
            c = rand(52)
            while c in used:
                c = rand(52)
            used.add(c)
            full.append(c)
        mine = _score(hero + full)
        beaten = False
        ties = 0
        for a, b in hands:
            s = _score([a, b] + full)
            if s > mine:
                beaten = True
                break
            if s == mine:
                ties += 1
        if not beaten:
            total += 1.0 / (ties + 1)
    return total / sims


# --------------------------------------------------------------------------
# Opponent statistics
# --------------------------------------------------------------------------

def _new_stats():
    return {
        # decisions, raises
        "dpre": 0, "rpre": 0, "dpost": 0, "rpost": 0,
        # times facing a raise, folds / re-raises when facing one
        "fpre": 0, "ffpre": 0, "frpre": 0,
        "fpost": 0, "ffpost": 0, "frpost": 0,
        # showdowns after they bet/raised: count, sum of hand percentiles
        "sn": 0, "ss": 0.0,
    }


_EMPTY = _new_stats()


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Model:

    def __init__(self):
        self.hand = []
        self.hand_ints = []
        self.player_id = None
        self.hands_played = 0
        self.rng = random.Random(20240607)
        self.stats = {}
        self.errors = 0
        self.last_error = ""
        self._nboard = 0
        self._last_raiser = None
        self._aggr = []
        self._shown = {}

    # -- engine hooks ------------------------------------------------------

    def seed(self, hand, player_id):
        self.hand = list(hand)
        self.hand_ints = [_card_int(c) for c in self.hand]
        self.player_id = player_id
        self.hands_played += 1
        self._nboard = 0
        self._last_raiser = None
        self._aggr = []
        self._shown = {}

    def run(self, actions, game_state):
        try:
            self._observe(actions, game_state)
        except Exception as error:
            if DEBUG:
                raise
            self.errors += 1
            self.last_error = "observe: " + repr(error)
        try:
            return self._act(game_state)
        except Exception as error:
            if DEBUG:
                raise
            self.errors += 1
            self.last_error = "act: " + repr(error)
            return self._fallback(game_state)

    def end_game(self, actions, game_state):
        try:
            self._observe(actions, game_state)
            teams = game_state.teams
            for p, cards in self._shown.items():
                if p == self.player_id or p not in self._aggr:
                    continue
                if len(cards) != 2:
                    continue
                s = self._stats_for(str(teams[p]))
                s["sn"] += 1
                s["ss"] += _class_pct(cards[0], cards[1])
        except Exception as error:
            if DEBUG:
                raise
            self.errors += 1
            self.last_error = "end_game: " + repr(error)

    # -- learning about opponents -----------------------------------------

    def _stats_for(self, name):
        s = self.stats.get(name)
        if s is None:
            s = _new_stats()
            self.stats[name] = s
        return s

    def _observe(self, actions, gs):
        me = self.player_id
        teams = gs.teams
        for a in actions:
            kind = a.kind
            p = a.player_id
            if kind == "reveal_card":
                if p == DEALER:
                    self._nboard += 1
                    self._last_raiser = None
                else:
                    self._shown.setdefault(p, []).append(_card_int(a.card))
            elif kind in ("fold", "call", "raise"):
                if p != me and 0 <= p < len(teams):
                    s = self._stats_for(str(teams[p]))
                    g = "pre" if self._nboard < 3 else "post"
                    s["d" + g] += 1
                    if kind == "raise":
                        s["r" + g] += 1
                    if self._last_raiser is not None and self._last_raiser != p:
                        s["f" + g] += 1
                        if kind == "fold":
                            s["ff" + g] += 1
                        elif kind == "raise":
                            s["fr" + g] += 1
                if kind == "raise":
                    self._last_raiser = p
                    if p not in self._aggr:
                        self._aggr.append(p)

    def _raise_freq(self, name, g):
        s = self.stats.get(name, _EMPTY)
        return (s["r" + g] + 0.35 * 6) / (s["d" + g] + 6.0)

    def _fold_freq(self, name, g):
        s = self.stats.get(name, _EMPTY)
        return (s["ff" + g] + 0.42 * 6) / (s["f" + g] + 6.0)

    def _reraise_freq(self, name, g):
        s = self.stats.get(name, _EMPTY)
        return (s["fr" + g] + 0.10 * 6) / (s["f" + g] + 6.0)

    def _range_fraction(self, name, g):
        """Fraction of all hands that this team's bets and raises come from."""
        s = self.stats.get(name, _EMPTY)
        q0 = _clamp(1.4 * self._raise_freq(name, g), 0.12, 1.0)
        q = (4.0 * q0 + 2.0 * s["ss"]) / (4.0 + s["sn"])
        return _clamp(q, 0.10, 1.0)

    # -- decisions -----------------------------------------------------------

    def _fallback(self, gs):
        me = self.player_id
        if gs.call_amount == 0:
            return PlayerAction.call(me)
        if gs.pot_odds < 0.2:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)

    def _opp_spec(self, p, name, hero, board, facing, call, P, tight, rng):
        """(mode, data) describing opponent p's range right now."""
        q = 1.0
        if p in self._aggr:
            g = "pre" if not board else "post"
            q = self._range_fraction(name, g)
            if facing:
                if q < 0.9:
                    size = call / max(P - call, 1.0)
                    q *= _clamp(1.15 - 0.3 * size, 0.6, 1.0)
            else:
                q = min(1.0, q * 1.35)
            q = max(0.08, q)
        if tight:
            q = max(0.06, q * 0.4)
        if not board:
            if q >= 0.97:
                return (0, None)
            return (1, _COMBOS[:max(12, int(q * 1326))])
        q = min(1.0, q * 1.25)
        if q >= 0.97:
            return (0, None)
        return (2, _cutoff(hero, board, q, rng))

    def _act(self, gs):
        me = self.player_id
        rng = self.rng
        hero = self.hand_ints
        board = [_card_int(c) for c in gs.board]
        pre = not board
        bb = gs.big_blind
        bets = gs.bets
        stacks = gs.stacks
        call = gs.call_amount
        cur = gs.current_bet
        my_bet = bets[me]
        stack = stacks[me]
        teams = gs.teams

        opps = [p for p in gs.players_in_hand if p != me]
        if not opps:
            return PlayerAction.call(me)
        live = [p for p in opps if stacks[p] > 0]
        names = {p: str(teams[p]) for p in opps}

        # Chips the opponent cannot match (when my call is all-in for less)
        # come back to them: they are not part of the pot I can win.
        excess = max(0, cur - my_bet - call)
        P = float(max(gs.pot - excess, 1))
        facing = call > 0
        can_raise = gs.can_raise

        sims = max(300, int(SIMS / max(1, len(opps)) ** 0.7))
        specs = [self._opp_spec(p, names[p], hero, board, facing, call, P,
                                False, rng) for p in opps]
        equity = _equity(hero, board, specs, sims, rng)

        g_now = "pre" if pre else "post"

        # ---- value of calling (or checking) --------------------------------
        if facing:
            no_more = (gs.street == "river" or call >= stack or not live)
            shade = 0.005 if no_more else 0.03
            e_call = max(0.0, equity - shade)
            ev_call = e_call * (P + call) - call
            if pre and cur <= bb:
                # Limping: the opponent may still raise and push me out.
                p_r = max(self._raise_freq(names[p], "pre") for p in opps)
                if e_call < 0.5:
                    ev_call = (1.0 - p_r) * ev_call + p_r * (-call)
        else:
            ev_call = max(0.0, equity - 0.01) * P

        # ---- value of raising ----------------------------------------------
        best_total = None
        best_ev = None
        if can_raise and live:
            lo, hi = gs.raise_range
            e_tight_specs = [self._opp_spec(p, names[p], hero, board, facing,
                                            call, P, True, rng)
                             for p in opps]
            e_tight = _equity(hero, board, e_tight_specs,
                              max(250, sims // 2), rng)

            cands = set()
            if pre and cur <= bb:
                for m in (2.2, 2.5, 3.0):
                    cands.add(int(m * bb))
            else:
                for fr in (0.33, 0.55, 0.75, 1.0):
                    cands.add(int(my_bet + call + fr * (P + call)))
            cands.add(hi)
            totals = set()
            for t in cands:
                t = min(hi, max(lo, t))
                x = t - my_bet
                if t < hi and hi - t <= 0.25 * (P + 2 * x):
                    t = hi  # nearly all my chips: commit
                totals.add(t)

            opp_cap = max(bets[p] + stacks[p] for p in opps)
            for t in sorted(totals):
                t_eff = min(t, opp_cap)
                if t_eff <= cur:
                    continue
                x = t_eff - my_bet
                add = t_eff - cur
                called_pot = P + x + add * len(live)
                b_frac = add / P
                mult = min(1.6, 0.65 + 0.6 * b_frac)
                f = 1.0
                r = 0.0
                for p in live:
                    fp = min(0.95, self._fold_freq(names[p], g_now) * mult)
                    if facing:
                        fp *= 0.75
                    f *= fp
                    if t < hi and t_eff < opp_cap:
                        r = max(r, self._reraise_freq(names[p], g_now))
                f = min(f, 0.95)
                r = min(r, max(0.0, 0.97 - f))
                # Equity when called: they call with the top (1 - f) of
                # their range, so interpolate toward my equity against a
                # tighter range.
                c = 1.0 - f
                e_c = e_tight + (equity - e_tight) * (c - 0.4) / 0.6
                e_c = _clamp(e_c, 0.0, 1.0)
                v_called = e_c * called_pot - x
                if equity < 0.55:
                    v_rr = -x
                else:
                    v_rr = (e_c - 0.12) * called_pot - x
                ev = f * P + r * v_rr + (1.0 - f - r) * v_called
                if best_ev is None or ev > best_ev:
                    best_ev = ev
                    best_total = t

        margin = 0.03 * P + 1.0

        if facing:
            if (best_total is not None and equity >= 0.48
                    and best_ev > max(ev_call, 0.0) + margin):
                return PlayerAction.raise_to(me, int(best_total))
            if ev_call > 0:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        if best_total is not None and best_ev > ev_call + margin:
            return PlayerAction.raise_to(me, int(best_total))
        return PlayerAction.call(me)