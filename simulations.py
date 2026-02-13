from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jax.scipy import stats
from plotly.subplots import make_subplots
from tqdm import tqdm


def random_player(rate: float):
    def _choice(rng_key, opponent_choice, prev_choice, params, even: bool = False):
        rate = params["rate"]
        return jax.random.binomial(rng_key, n=1, p=rate), params

    return _choice, {"rate": rate}


def wsls_player():
    def _choice(rng_key, opponent_choice, prev_choice, params, even: bool = False):
        did_win = (
            opponent_choice == prev_choice if even else opponent_choice != prev_choice
        )
        return jnp.where(did_win, prev_choice, 1 - prev_choice), params

    return _choice, {}


def loose_streak_wsls_player(buffer_len: int):
    def _choice(
        rng_key,
        opponent_choice,
        prev_choice,
        params,
        even: bool = False,
    ):
        did_win = (
            opponent_choice == prev_choice if even else opponent_choice != prev_choice
        )
        streak = jnp.append(params["streak_buffer"], did_win)[1:]
        switch_prob = 1 - (1 / (buffer_len + 1) * jnp.sum(streak))
        switch = jax.random.binomial(rng_key, n=1, p=switch_prob)
        choice = jnp.where(switch, prev_choice, 1 - prev_choice)
        return choice, {"streak_buffer": streak}

    return _choice, {"streak_buffer": jnp.ones(buffer_len)}


def beta_binomial_player(alpha: float, beta: float):
    def _choice(
        rng_key,
        opponent_choice,
        prev_choice,
        params,
        even: bool = False,
    ):
        # posterior: Beta(y+alpha, 1-y+beta)
        alpha = params["alpha"] + opponent_choice
        beta = params["beta"] + 1 - opponent_choice
        rng_key, subkey = jax.random.split(rng_key)
        opponent_p = jax.random.beta(subkey, alpha, beta)
        p = opponent_p if even else 1 - opponent_p
        rng_key, subkey = jax.random.split(rng_key)
        choice = jax.random.binomial(subkey, n=1, p=p)

        return choice, {"alpha": alpha, "beta": beta}

    return _choice, {"alpha": alpha, "beta": beta}


def deterministic_beta_player(alpha: float, beta: float):
    def _choice(
        rng_key,
        opponent_choice,
        prev_choice,
        params,
        even: bool = False,
    ):
        # posterior: Beta(y+alpha, 1-y+beta)
        alpha = params["alpha"] + opponent_choice
        beta = params["beta"] + 1 - opponent_choice
        is_deterministic = (alpha > 1) & (beta > 1)
        random_choice = jax.random.binomial(rng_key, n=1, p=0.5)
        p_mode = jnp.where(is_deterministic, (alpha - 1) / (alpha + beta - 2), 0.5)
        determ_choice = jnp.where(p_mode > 0.5, 1, 0)
        choice = jnp.where(is_deterministic, determ_choice, random_choice)
        return choice, {"alpha": alpha, "beta": beta}

    return _choice, {"alpha": alpha, "beta": beta}


def play_game(rng_key, players: list[tuple[Callable, dict]], n_trials: int = 120):
    (odd_choice_fn, odd_params), (even_choice_fn, even_params) = players
    even_choice_fn = partial(even_choice_fn, even=True)
    odd_choice_fn = partial(odd_choice_fn, even=False)

    @jax.jit
    def step(state, rng_key):
        odd_params = state["odd_params"]
        even_params = state["even_params"]
        odd_choice = state["odd_choice"]
        even_choice = state["even_choice"]
        even_key, odd_key = jax.random.split(rng_key)
        new_even_choice, new_even_params = even_choice_fn(
            even_key, odd_choice, even_choice, even_params
        )
        new_odd_choice, new_odd_params = odd_choice_fn(
            odd_key, even_choice, odd_choice, odd_params
        )
        new_state = {
            "even_params": new_even_params,
            "odd_params": new_odd_params,
            "even_choice": new_even_choice,
            "odd_choice": new_odd_choice,
        }
        return new_state, new_state

    keys = jax.random.split(rng_key, n_trials)
    init_state = {
        "even_params": even_params,
        "odd_params": odd_params,
        "even_choice": 0,
        "odd_choice": 0,
    }
    last_state, states = jax.lax.scan(step, init_state, keys)
    return states


def play_games(
    rng_key,
    players: list[tuple[Callable, dict]],
    n_trials: int = 120,
    n_games: int = 100,
):
    keys = jax.random.split(rng_key, n_games)
    return jax.vmap(partial(play_game, players=players, n_trials=n_trials))(keys)


def plot_bayesian_updates(states, n_grid_points=100):
    fig = go.Figure()
    colors = px.colors.sample_colorscale("thermal", np.arange(n_trials) / n_trials)
    grid = jnp.linspace(0, 1.0, n_grid_points)
    for alpha, beta, color in zip(
        states["even_params"]["alpha"], states["even_params"]["beta"], colors
    ):
        fig.add_scatter(
            x=grid,
            y=stats.beta.pdf(grid, a=alpha, b=beta),
            line=dict(color=color),
            showlegend=False,
        )
    fig.update_layout(template="plotly_white")
    return fig


def plot_win_rate(states):
    even_wins = states["odd_choice"] == states["even_choice"]
    n_games, n_trials = even_wins.shape
    win_rate = jnp.cumsum(even_wins, axis=1) / (jnp.arange(n_trials) + 1)[None, :]
    mean_wr = jnp.mean(win_rate, axis=0)
    lower_wr, upper_wr = np.quantile(win_rate, [0.05, 0.95], axis=0)
    x = jnp.arange(n_trials)
    fig = go.Figure()
    fig = fig.add_scatter(
        x=x, y=mean_wr, line=dict(color="rgb(0,100,80)"), mode="lines"
    )
    fig = fig.add_scatter(
        name="Upper Bound",
        x=x,
        y=upper_wr,
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False,
    )
    fig = fig.add_scatter(
        name="Lower Bound",
        x=x,
        y=lower_wr,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode="lines",
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
        showlegend=False,
    )
    fig.update_layout(template="plotly_white")
    return fig


def take(pytree, index):
    # Indexing a pytree along axis 0
    leaves, treedef = jax.tree.flatten(states)
    leaves = [leaf[index] for leaf in leaves]
    return jax.tree.unflatten(treedef, leaves)


n_trials = 120
n_games = 100
rng_key = jax.random.key(0)
states = jax.vmap(play_game)
states = play_games(
    rng_key,
    players=[random_player(0.6), deterministic_beta_player(alpha=1.0, beta=10.0)],
    n_trials=n_trials,
    n_games=n_games,
)


fig = make_subplots(rows=10, cols=10, horizontal_spacing=0, vertical_spacing=0)
for i in range(n_games):
    subfig = plot_bayesian_updates(take(states, i))
    row, col = (i // 10) + 1, (i % 10) + 1
    for trace in subfig.data:
        fig.add_trace(trace, row=row, col=col)
fig = fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), template="plotly_white")
fig.show()

states.keys()

plot_win_rate(states)


jnp.sum(states["odd_choice"] == states["even_choice"])

fig = go.Figure()
fig = fig.add_scatter(x=jnp.arange(120), y=states["odd_choice"], name="Odd")
fig = fig.add_scatter(x=jnp.arange(120), y=states["even_choice"], name="Even")
fig.show()

even_wins = states["even_choice"] == states["odd_choice"]
even_win_rate = jnp.cumsum(even_wins) / (jnp.arange(len(even_wins)) + 1)
px.line(x=jnp.arange(len(even_wins)), y=even_win_rate)

even_choices
