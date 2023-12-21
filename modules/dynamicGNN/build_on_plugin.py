import importlib


def import_plugin_model(model):
    module = importlib.import_module("modules.plugins." + model)
    return getattr(module, model)

def import_gnn_model(model):
    module = importlib.import_module("modules.dynamicGNN." + model)
    return getattr(module, model)

def generate_plugin_dynamicGNN(plugin_model, gnn_model):
    pass