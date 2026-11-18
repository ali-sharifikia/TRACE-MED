from runtime_resources import resource as asp_resource
NAMESPACE = 'specialist_v1_26'

def verify_contract(*args, **kwargs):
    raise ValueError('Unbound experimental authority is not supported in ASP-MT')

def expanded_selection_authorized(*args, **kwargs):
    return False
