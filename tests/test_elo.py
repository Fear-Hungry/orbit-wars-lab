from python.league.elo import EloRating, update_elo


def test_update_elo_win():
    a, b = update_elo(EloRating(), EloRating(), 1.0)
    assert a.rating > 1000
    assert b.rating < 1000
