"""Old PVO snapshot reuse, zero calculator work."""
import importlib.util
from pathlib import Path
import sys
from dataclasses import replace

import pytest

from xtbflow.calculators import CalculationResult, MolecularSystem
from xtbflow.sampling import GenerationSnapshot, GuidanceLane, ProviderRequest

spec = importlib.util.spec_from_file_location('live_pvo', Path(__file__).parents[1]/'scripts/run_live_matched_pvo.py')
pvo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pvo
spec.loader.exec_module(pvo)


@pytest.mark.parametrize('stale', [False, True])
def test_reused_provider_rebuilds_payload_and_rejects_stale_hash(stale):
    original = MolecularSystem(('H', 'H'), ((0., 0., 0.), (1., 0., 0.)), 0, 1, {'periodic': False})
    seen = []
    class Backend:
        def evaluate(self, current, **kwargs):
            seen.append(current)
            return CalculationResult(calculator='xtb_oracle', protocol_id='fake',
                input_hash=original.input_hash if stale else current.input_hash,
                charge=current.charge, multiplicity=current.multiplicity, operation='energy_forces',
                energy=-1., forces=((.1, 0., 0.), (-.1, 0., 0.)))
    provider = pvo.LivePhysicalProvider(original, Backend, pvo.CalculatorCallCounter(2))
    first = GenerationSnapshot('p', 'a', 0, original.coordinates, (0., 0.), (True, True), 0,
                               metadata={'input_hash': original.input_hash})
    second = replace(first, generation_step=1, geometry=((0.,0.,0.), (1.3,0.,0.)), snapshot_id='')
    def request(snapshot):
        return ProviderRequest(snapshot, GuidanceLane.GEOMETRY, provider.provider_revision,
                               provider.config_fingerprint, provider.protocol_fingerprint)
    provider.invoke(request(first))
    if stale:
        with pytest.raises(ValueError, match='current snapshot'):
            provider.invoke(request(second))
    else:
        output = provider.invoke(request(second))
        physical = output.to_mapping()['provenance']['physical']
        assert physical['original_input_hash'] == original.input_hash
        assert physical['calculator_input_hash'] == seen[1].input_hash != original.input_hash
        assert output.input_fingerprint == second.snapshot_id
    assert seen[1].coordinates == second.geometry
    assert seen[1].symbols == original.symbols
    assert seen[1].environment == original.environment
    assert provider.call_counter.count == 2
