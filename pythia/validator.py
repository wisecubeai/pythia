import yaml
import os


def load_config():
    try:
        # for local test use default
        config_file = os.getenv("CONFIG_FILE", default="../configurations/validators/config.yaml")
        with open(config_file) as f:
            config = yaml.safe_load(f)
        print("load file {}".format(config_file))
        return config
    except Exception as e:
        print(e)
    return None


class ValidatorPool():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            validators = load_config()
            cls._instance = super(ValidatorPool, cls).__new__(cls)
            cls._instance._init(validators)
        return cls._instance

    def _init(self, validators):
        self._enabled_validators = []
        if validators is not None:
            for v_conf in validators["validators"]["options"]:
                validator = Validator(v_conf)
                if validator.enable:
                    print("Adding validator {} to the pool".format(validator.name))
                    self._enabled_validators.append(validator.name)

    @property
    def enabled_validators(self):
        return self._enabled_validators


class Validator():
    def __init__(self, config):
        self.name = config["name"]
        self.enable = config["enable"]
        self.description = config["description"]
