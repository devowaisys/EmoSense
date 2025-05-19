import configparser
import psycopg2

class DatabaseConfig:
    def __init__(self, config_file=r"config.ini"):
        self.config_file = config_file
        self.config = self.read_config()

    def read_config(self):
        """Read the configuration file and return the database configuration."""
        config = configparser.ConfigParser()
        config.read(self.config_file)

        if 'database' not in config:
            raise KeyError("Missing 'database' section in config.ini")

        return {
            "host": config['database']['host'],
            "port": config['database']['port'],
            "user": config['database']['user'],
            "password": config['database']['password'],
            "database": config['database']['database']
        }

    def get_connection(self):
        """Establish and return a database connection."""
        return psycopg2.connect(**self.config)
