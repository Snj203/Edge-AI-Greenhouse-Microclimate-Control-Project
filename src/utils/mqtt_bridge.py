"""JSON publish/subscribe over the local Mosquitto MQTT broker (paho-mqtt)."""

import json
import threading
import uuid
from datetime import datetime
from typing import Callable

import paho.mqtt.client as mqtt

from src.utils.config import load_config


# Connection to the MQTT broker: send and receive JSON messages
class MQTTBridge:
    # Read the broker address and topics from the config, create the paho client
    def __init__(self, config: dict | None = None, client_name: str = "greenhouse"):
        cfg = (config or load_config())["mqtt"]
        self.broker_host = cfg["broker_host"]
        self.broker_port = cfg["broker_port"]
        self.topic_sensors = cfg["topic_sensors"]
        self.topic_actuators = cfg["topic_actuators"]

        self._callbacks: dict[str, Callable[[dict], None]] = {}
        self._connected = threading.Event()
        self._connect_error: str | None = None

        self._client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                                   client_id=f"{client_name}-{uuid.uuid4().hex[:8]}")
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    # paho calls these from its own network thread

    # Connected: subscribe (again) to every registered topic
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self._connect_error = str(reason_code)
            return
        self._connect_error = None
        for topic in self._callbacks:  # subscribe again after a reconnect
            client.subscribe(topic)
        self._connected.set()

    # Connection lost
    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self._connected.clear()

    # Message received: decode the JSON and pass it to the callback of its topic
    def _on_message(self, client, userdata, msg):
        callback = self._callbacks.get(msg.topic)
        if callback is None:
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[MQTT] invalid JSON on {msg.topic}: {e}")
            return
        try:
            callback(payload)
        except Exception as e:  # an error in a callback must not stop the network thread
            print(f"[MQTT] callback error on {msg.topic}: {e!r}")

    # Connect to the broker and start the network thread
    def connect(self, timeout: float = 5.0) -> "MQTTBridge":
        try:
            self._client.connect(self.broker_host, self.broker_port, keepalive=60)
        except OSError as e:
            raise ConnectionError(f"MQTT connection to {self.broker_host}:{self.broker_port} failed: {e}. "
                                  "Is Mosquitto running?") from e
        self._client.loop_start()
        if not self._connected.wait(timeout):
            self._client.loop_stop()
            raise ConnectionError(f"MQTT broker refused or timed out: {self._connect_error or 'no answer'}")
        return self

    # Disconnect and stop the network thread
    def disconnect(self):
        self._client.disconnect()
        self._client.loop_stop()
        self._connected.clear()

    # `with MQTTBridge(...) as bridge:` connects at the start...
    def __enter__(self):
        return self.connect()

    # ...and disconnects at the end
    def __exit__(self, *exc):
        self.disconnect()

    # Send `data` as JSON with a timestamp (now, unless given)
    def publish(self, topic: str, data: dict, timestamp: str | None = None, wait: bool = False) -> dict:
        payload = {"timestamp": timestamp or datetime.now().isoformat(timespec="seconds"), **data}
        info = self._client.publish(topic, json.dumps(payload), qos=0)
        if wait:
            info.wait_for_publish(timeout=2.0)
        return payload

    # Call `callback(dict)` for every JSON message on `topic`
    def subscribe(self, topic: str, callback: Callable[[dict], None]):
        self._callbacks[topic] = callback
        if self._connected.is_set():
            self._client.subscribe(topic)

    # Send sensor readings
    def publish_sensors(self, data: dict, timestamp: str | None = None):
        return self.publish(self.topic_sensors, data, timestamp)

    # Send an actuator command
    def publish_actuators(self, data: dict, timestamp: str | None = None):
        return self.publish(self.topic_actuators, data, timestamp)

    # Listen to sensor readings
    def subscribe_sensors(self, callback: Callable[[dict], None]):
        self.subscribe(self.topic_sensors, callback)

    # Listen to actuator commands
    def subscribe_actuators(self, callback: Callable[[dict], None]):
        self.subscribe(self.topic_actuators, callback)
