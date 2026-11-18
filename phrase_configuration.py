from runtime_resources import resource as asp_resource
import phrase_evidence_consistency as original
import phrase_normalization as phrases
method = original.previous.loader.isolated('_asp_phrase_configuration', original.__file__)
method.phrases = phrases
method.PHRASE_SOURCE = phrases.DEFAULT_SOURCE
method.default_normalizer = phrases.default_normalizer
