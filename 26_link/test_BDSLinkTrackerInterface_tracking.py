# pip install pytest-xdist
# pytest -n auto

import os
from pathlib import Path
from typing import Callable

import pytest

pytestmark = pytest.mark.xfail(reason="requires bdsim")


def drift_proton_5TeV(batch=1) :
    import bdsim
    import numpy as np

    def polarToCartesianMomenta(theta, phi, momentum):
        px = momentum*np.sin(theta)*np.cos(phi)
        py = momentum*np.sin(theta)*np.sin(phi)
        pz = momentum*np.cos(theta)

        return [float(px), float(py), float(pz)]

    tracker_l = bdsim.BDSLinkTrackerInterface.GetInstance("./trackerInterface.gmad", 2212, 5*bdsim.clhep.TeV, 0.01, 1234, 1, batch)
    bdsim_l = tracker_l.GetBDSIMLink()
    bunch_l = tracker_l.GetBunchLink()

    # create element
    e = bdsim.Element()
    e.name = "drift1"
    e.type = bdsim.elementtype.ElementType.DRIFT
    e['l'] = 1.0

    # add element
    bdsim_l = tracker_l.GetBDSIMLink()
    bdsim_l.AddLinkElement(e)

    # select element
    bdsim_l.SelectLinkElement("drift1")

    # reference particle
    rp = tracker_l.GetReferenceParticleDefinition()

    # number of particles
    ngenerate = 1000

    # add particle

    rng = np.random.default_rng(seed=1234)
    theta = rng.uniform(0, 1e-8, ngenerate)
    phi = rng.uniform(0, 2*np.pi,ngenerate)
    xmm = rng.uniform(-1, 1, ngenerate)
    ymm = rng.uniform(-1, 1, ngenerate)

    for i, [t,p,x,y] in enumerate(zip(theta, phi, xmm,ymm)) :
        mom = polarToCartesianMomenta(t, p, rp.Momentum())
        tracker_l.AddParticle(x, y,  *mom , 0.,0.,  i,2212)  # pdgid;

    # beam on
    bdsim_l.BeamOn(ngenerate)

    # do not perform analysis if not in batch
    if batch == 0:
        return 0

    # get sampler data
    shs = bdsim_l.SamplerHits()

    dx = []
    dy = []

    # check output compared to input
    for i in range(0, bunch_l.Size()) :
        bunch_l.SetCurrentIndex(i)
        pl = bunch_l.ParticleLocal()
        sh = shs[i]

        xSlope = pl.xp/pl.zp
        ySlope = pl.yp/pl.zp

        dx.append(sh.coords.x - (pl.x + xSlope*1000))
        dy.append(sh.coords.y - (pl.y + ySlope*1000))

    dx = np.array(dx)
    dy = np.array(dy)
    assert dx.std() < 1e-11
    assert dy.std() < 1e-11

    if "PYTEST_CURRENT_TEST" in os.environ:
        tracker_l.Reset()
        return 0
    else :
        return tracker_l

def collimator_proton_5TeV(batch=1) :
    import bdsim
    import pybdsim
    import numpy as np

    def polarToCartesianMomenta(theta, phi, momentum):
        px = momentum*np.sin(theta)*np.cos(phi)
        py = momentum*np.sin(theta)*np.sin(phi)
        pz = momentum*np.cos(theta)

        return [float(px), float(py), float(pz)]

    tracker_l = bdsim.BDSLinkTrackerInterface.GetInstance("./trackerInterface.gmad", 2212, 5*bdsim.clhep.TeV, 0.01, 1234, 1, batch)
    bdsim_l = tracker_l.GetBDSIMLink()
    bunch_l = tracker_l.GetBunchLink()

    # create element
    e = bdsim.Element()
    e.name = "rcol1"
    e.type = bdsim.elementtype.ElementType.RCOL
    e['material'] = "G4_Fe"
    e['l'] = 0.1
    e['xsize'] = 0.0
    e['ysize'] = 0.0

    # add element
    bdsim_l = tracker_l.GetBDSIMLink()
    bdsim_l.AddLinkElement(e)

    # select element
    bdsim_l.SelectLinkElement("rcol1")

    # reference particle
    rp = tracker_l.GetReferenceParticleDefinition()

    # number of particles
    ngenerate = 1000

    # add particle

    for i in range(0,ngenerate) :
        tracker_l.AddParticle(0, 0,  *[0,0, rp.Momentum()] , 0.,0.,  i,2212)  # pdgid;

    # beam on
    bdsim_l.BeamOn(ngenerate)

    # do not perform analysis if not in batch
    if batch == 0:
        return 0

    # get sampler data
    shs = bdsim_l.SamplerHits()

    # make dataframe
    df = pybdsim.DataPandas.LinkSamplerHits(shs).get_dataframe()

    if "PYTEST_CURRENT_TEST" in os.environ:
        tracker_l.Reset()
        return 0
    else :
        return tracker_l, df

def quadrupole_thick_matrix() -> int:
    import bdsim
    import math

    length = 0.5       # m
    k1 = 0.2           # m^-2
    x0 = 10.0          # mm
    y0 = 2.0           # mm

    parser = bdsim.BDSParser()
    beam = parser.GetBeam()
    beam["particleName"] = "proton"
    beam["kineticEnergy"] = 100.0  # GeV

    quadrupole = bdsim.Element()
    quadrupole.name = "quadrupole"
    quadrupole.type = bdsim.elementtype.ElementType.QUAD
    quadrupole["l"] = length
    quadrupole["k1"] = k1
    quadrupole["apertureType"] = "circular"
    quadrupole["aper1"] = 0.1
    quadrupole["outerDiameter"] = 0.3
    parser.GetBeamline().push_back(quadrupole, False, "element")

    tracker = bdsim.BDSLinkTrackerInterface.GetInstance(
        parser,
        referenceParticlePDG=2212,
        referenceKineticEnergy=100 * bdsim.clhep.GeV,
    )
    tracker.AddParticle(x0, y0, 0, 0, 0, 0, 1, 1, 0, 17, 2212)

    link = tracker.GetBDSIMLink()
    link.SelectLinkElement("quadrupole")
    link.BeamOn(1)

    hits = link.SamplerHits()
    assert hits.entries() == 1
    coordinates = hits[0].coords

    root_k = math.sqrt(k1)
    phase = root_k * length
    expected = {
        "x": math.cos(phase) * x0,
        "xp": -root_k * math.sin(phase) * x0 * 1e-3,
        "y": math.cosh(phase) * y0,
        "yp": root_k * math.sinh(phase) * y0 * 1e-3,
    }

    assert coordinates.x == pytest.approx(expected["x"], abs=2e-5)
    assert coordinates.xp == pytest.approx(expected["xp"], abs=2e-8)
    assert coordinates.y == pytest.approx(expected["y"], abs=2e-5)
    assert coordinates.yp == pytest.approx(expected["yp"], abs=2e-8)

    return 0

def test_drift_proton_5TeV(make_bdsim_test_code, run_bdsim_test_code_as_subprocess) :
    os.chdir(Path(__file__).resolve().parent)
    code = make_bdsim_test_code(drift_proton_5TeV, args="", dir=os.path.dirname(os.path.abspath(__file__)))
    result = run_bdsim_test_code_as_subprocess(code)

def test_quadrupole_thick_matrix(
    make_bdsim_test_code: Callable[..., str],
    run_bdsim_test_code_as_subprocess: Callable[[str], str],
) -> None:
    code = make_bdsim_test_code(
        quadrupole_thick_matrix,
        args="",
        dir=os.path.dirname(os.path.abspath(__file__)),
    )
    run_bdsim_test_code_as_subprocess(code)

# TODO
#def test_collimator_proton_5TeV(make_bdsim_test_code, run_bdsim_test_code_as_subprocess) :
#    os.chdir(Path(__file__).resolve().parent)
#    code = make_bdsim_test_code(collimator_proton_5TeV, args="", dir=os.path.dirname(os.path.abspath(__file__)))
#    result = run_bdsim_test_code_as_subprocess(code)
