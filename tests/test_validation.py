"""Input validation tests for the public classes."""

import numpy as np
import pytest

import pysemi
from pysemi.sgd import LinearRegressionOracle


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_semi_star_rejects_bad_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        pysemi.SeMIStar(mu=0.0, sigma=sigma, z_star=1.0, n=10)


@pytest.mark.parametrize("n", [0, 1, -3])
def test_semi_star_rejects_bad_n(n):
    with pytest.raises(ValueError, match="n"):
        pysemi.SeMIStar(mu=0.0, sigma=1.0, z_star=1.0, n=n)


@pytest.mark.parametrize("eps", [0.0, -0.1, np.nan, np.inf])
def test_semi_star_dp_rejects_bad_epsilon(eps):
    with pytest.raises(ValueError, match="epsilon"):
        pysemi.SeMIStarDP(mu=0.0, sigma=1.0, z_star=1.0, n=10, epsilon=eps)


def test_semi_star_dp_rejects_bad_sensitivity():
    with pytest.raises(ValueError, match="sensitivity"):
        pysemi.SeMIStarDP(mu=0.0, sigma=1.0, z_star=1.0, n=10, epsilon=1.0, sensitivity=-1.0)


def test_semi_star_dp_power_raises():
    test = pysemi.SeMIStarDP(mu=0.0, sigma=1.0, z_star=1.0, n=10, epsilon=1.0)
    with pytest.raises(NotImplementedError, match="empirical"):
        test.power(alpha=0.05, T=10, tau=5)


def test_final_observation_rejects_bad_T():
    with pytest.raises(ValueError, match="T"):
        pysemi.FinalObservation(mu=0.0, sigma=1.0, z_star=1.0, n=10, T=0)


def test_semi_sgd_rejects_mismatched_shapes():
    oracle = LinearRegressionOracle()
    thetas = [np.zeros(3), np.zeros(3), np.zeros(3)]
    # mu_g has wrong dim
    with pytest.raises(ValueError, match="mu_g"):
        pysemi.SeMISGD(
            thetas, oracle, mu_g=np.zeros(2), sigma_inv=np.eye(3),
            eta=0.1, n_batch=10,
        )
    # sigma_inv has wrong dim
    with pytest.raises(ValueError, match="sigma_inv"):
        pysemi.SeMISGD(
            thetas, oracle, mu_g=np.zeros(3), sigma_inv=np.eye(2),
            eta=0.1, n_batch=10,
        )


def test_semi_sgd_rejects_inconsistent_trajectory():
    oracle = LinearRegressionOracle()
    thetas = [np.zeros(3), np.zeros(4)]  # mismatched
    with pytest.raises(ValueError, match="theta_trajectory"):
        pysemi.SeMISGD(
            thetas, oracle, mu_g=np.zeros(3), sigma_inv=np.eye(3),
            eta=0.1, n_batch=10,
        )


@pytest.mark.parametrize("eta", [0.0, -0.1, np.nan, np.inf])
def test_semi_sgd_rejects_bad_eta(eta):
    oracle = LinearRegressionOracle()
    with pytest.raises(ValueError, match="eta"):
        pysemi.SeMISGD(
            [np.zeros(3), np.zeros(3)], oracle, mu_g=np.zeros(3), sigma_inv=np.eye(3),
            eta=eta, n_batch=10,
        )


def test_repr_is_informative():
    assert "SeMIStar" in repr(pysemi.SeMIStar(0.0, 1.0, 2.0, 10))
    assert "epsilon" in repr(pysemi.SeMIStarDP(0.0, 1.0, 2.0, 10, epsilon=1.0))
    assert "FinalObservation" in repr(pysemi.FinalObservation(0.0, 1.0, 2.0, 10, 5))
    assert "SeMISGD" in repr(pysemi.SeMISGD(
        [np.zeros(3), np.zeros(3)], LinearRegressionOracle(),
        np.zeros(3), np.eye(3), eta=0.1, n_batch=10,
    ))
