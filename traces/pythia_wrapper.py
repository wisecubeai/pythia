import openlit


class PythiaTraces:
    def __init__(self, endpoint, service_name="default"):
        self.endpoint = endpoint
        self.service_name = service_name
        self.initialized = False

    def init(self):
        try:
            openlit.init(otlp_endpoint=self.endpoint, application_name=self.service_name)
            self.initialized = True
            print(f"PythiaTraces initialized with endpoint {self.endpoint}")
        except Exception as e:
            print(f"Failed to initialize PythiaTraces: {e}")
            self.initialized = False

    def is_initialized(self):
        return self.initialize
