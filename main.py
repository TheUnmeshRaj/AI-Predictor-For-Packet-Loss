import logging
import os
import queue
import socket
import threading
import time
from collections import deque
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ping3
import plotly.graph_objs as go
import psutil
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkMonitor:
    def __init__(self, target_ip="172.17.0.2"):  
        self.target_ip = target_ip
        self.rtt_history = deque(maxlen=100)
        self.jitter_history = deque(maxlen=100)
        self.bandwidth_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.packet_loss_queue = queue.Queue()
        self.packets_sent = 0
        self.packets_received = 0

    def measure_rtt(self):
        try:
            rtt = ping3.ping(self.target_ip, unit="ms", timeout=1)
            if rtt is None:
                rtt = 1000  
            return rtt
        except Exception as e:
            logger.warning(f"RTT measurement failed: {e}")
            return 1000

    def measure_bandwidth(self):
        try:
            net_io = psutil.net_io_counters()
            return net_io.bytes_sent + net_io.bytes_recv
        except Exception as e:
            logger.warning(f"Bandwidth measurement failed: {e}")
            return 0

    def calculate_jitter(self, rtt):
        self.rtt_history.append(rtt)
        if len(self.rtt_history) > 1:
            return np.std(list(self.rtt_history))
        return 0

    def measure_packet_loss(self):
        self.packets_sent += 100
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)
            for _ in range(100):
                sock.sendto(b"test", (self.target_ip, 6001))
                try:
                    sock.recvfrom(1024)
                    self.packets_received += 1
                except socket.timeout:
                    pass
            sock.close()
        except Exception as e:
            logger.warning(f"Packet loss measurement failed: {e}")
        loss_percentage = ((self.packets_sent - self.packets_received) / self.packets_sent) * 100
        self.loss_history.append(loss_percentage)
        return loss_percentage

    def collect_metrics(self):
        rtt = self.measure_rtt()
        jitter = self.calculate_jitter(rtt)
        bandwidth = self.measure_bandwidth()
        loss = self.measure_packet_loss()
        metrics = {
            "rtt": rtt,
            "jitter": jitter,
            "bandwidth": bandwidth,
            "loss": loss
        }
        self.packet_loss_queue.put(metrics)
        return metrics


class PacketLossPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def generate_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        data = {
            "rtt": np.random.normal(50, 20, n_samples),
            "jitter": np.random.normal(5, 2, n_samples),
            "bandwidth": np.random.normal(1000000, 200000, n_samples),
            "past_loss": np.random.uniform(0, 20, n_samples)
        }
        df = pd.DataFrame(data)
        df["loss_next"] = (df["past_loss"] + np.random.normal(0, 5, n_samples) > 5).astype(int)
        return df

    def train(self):
        try:
            data = self.generate_synthetic_data()
            X = data[["rtt", "jitter", "bandwidth", "past_loss"]]
            y = data["loss_next"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            logger.info(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            logger.info(f"Model F1-Score: {f1_score(y_test, y_pred):.2f}")
            self.is_trained = True
        except Exception as e:
            logger.error(f"Model training failed: {e}")

    def predict(self, metrics):
        if not self.is_trained:
            self.train()
        try:
            features = pd.DataFrame([[metrics["rtt"], metrics["jitter"], metrics["bandwidth"], metrics["loss"]]], 
                                   columns=["rtt", "jitter", "bandwidth", "past_loss"])
            return self.model.predict(features)[0]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0


class VideoCallSimulator:
    def __init__(self, host="172.17.0.2", port=6000):
        self.host = host
        self.port = port
        self.resolution = (640, 480)  
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.running = False
        self.cap = cv2.VideoCapture(0)  
        
        
        self.monitor = NetworkMonitor()
        self.predictor = PacketLossPredictor()

    def adjust_quality(self, predicted_loss):
        if predicted_loss == 1:
            self.resolution = (320, 240)  
            logger.info("Reducing resolution to 240p due to predicted packet loss")
        else:
            self.resolution = (640, 480)  
            logger.info("Restoring resolution to 480p")

    def send_frame(self, frame):
        try:
            frame = cv2.resize(frame, self.resolution)
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if len(buffer) > 65000:
                logger.warning(f"Buffer size {len(buffer)} exceeds UDP limit, skipping send")
                return
            self.sock.sendto(buffer, (self.host, self.port))
        except Exception as e:
            logger.warning(f"Failed to send frame: {e}")

    def receive_frame(self):
        try:
            data, addr = self.sock.recvfrom(65507)
            if not data:
                logger.warning("No data received")
                return None
            logger.debug(f"Received {len(data)} bytes from {addr}")
            np_data = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Failed to decode frame")
                return None
            return frame
        except socket.timeout:
            logger.debug("Receive timeout")
            return None
        except Exception as e:
            logger.warning(f"Error receiving frame: {e}")
            return None
    def run_client_a(self):
        self.running = True
        if not self.cap.isOpened():
            logger.error("Failed to open video capture")
            return
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to capture frame")
                continue
            metrics = self.monitor.collect_metrics()
            predicted_loss = self.predictor.predict(metrics)
            self.adjust_quality(predicted_loss)
            self.send_frame(frame)
            try:
                cv2.imshow("Client A - Sending", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error as e:
                logger.error(f"OpenCV display error in Client A: {e}")
                continue
        self.cleanup()

    def run_client_b(self):
        self.running = True
        try:
            self.sock.bind(("0.0.0.0", self.port))
        except OSError as e:
            logger.error(f"Failed to bind socket: {e}")
            return
        while self.running:
            frame = self.receive_frame()
            if frame is not None and frame.size > 0:
                try:
                    cv2.imshow("Client B - Receiving", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error as e:
                    logger.error(f"OpenCV display error in Client B: {e}")
                    continue
            else:
                logger.warning("Received invalid or no frame")
            time.sleep(0.01)  
        self.cleanup()

    def cleanup(self):
        self.running = False
        try:
            self.cap.release()
        except:
            pass
        try:
            self.sock.close()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass


def run_streamlit_dashboard():
    st.title("AI Packet Loss Predictor Dashboard")
    st.markdown("Real-time network metrics and packet loss predictions")

    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []

    monitor = NetworkMonitor()
    chart_rtt = st.empty()
    chart_jitter = st.empty()
    chart_loss = st.empty()
    prediction_text = st.empty()

    while True:
        try:
            metrics = monitor.collect_metrics()
            st.session_state.metrics_history.append(metrics)
            if len(st.session_state.metrics_history) > 100:
                st.session_state.metrics_history.pop(0)

            times = list(range(len(st.session_state.metrics_history)))
            rtt_data = [m["rtt"] for m in st.session_state.metrics_history]
            jitter_data = [m["jitter"] for m in st.session_state.metrics_history]
            loss_data = [m["loss"] for m in st.session_state.metrics_history]

            fig_rtt = go.Figure()
            fig_rtt.add_trace(go.Scatter(x=times, y=rtt_data, mode='lines', name='RTT (ms)'))
            fig_rtt.update_layout(title="Round-Trip Time", xaxis_title="Time", yaxis_title="RTT (ms)")
            chart_rtt.plotly_chart(fig_rtt, use_container_width=True)

            fig_jitter = go.Figure()
            fig_jitter.add_trace(go.Scatter(x=times, y=jitter_data, mode='lines', name='Jitter (ms)'))
            fig_jitter.update_layout(title="Jitter", xaxis_title="Time", yaxis_title="Jitter (ms)")
            chart_jitter.plotly_chart(fig_jitter, use_container_width=True)

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=times, y=loss_data, mode='lines', name='Packet Loss (%)'))
            fig_loss.update_layout(title="Packet Loss", xaxis_title="Time", yaxis_title="Packet Loss (%)")
            chart_loss.plotly_chart(fig_loss, use_container_width=True)

            predictor = PacketLossPredictor()
            prediction = predictor.predict(metrics)
            prediction_text.write(f"Predicted Packet Loss: {'High' if prediction == 1 else 'Low'}")

            time.sleep(1)
        except Exception as e:
            logger.error(f"Streamlit dashboard error: {e}")
            time.sleep(1)


def plot_metrics(metrics_history):
    try:
        rtt = [m["rtt"] for m in metrics_history]
        jitter = [m["jitter"] for m in metrics_history]
        loss = [m["loss"] for m in metrics_history]
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(rtt, label="RTT (ms)")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(jitter, label="Jitter (ms)")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(loss, label="Packet Loss (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("metrics_plot.png")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot metrics: {e}")

def main():
    simulator = VideoCallSimulator()
    metrics_history = []
    
    
    client_a_thread = threading.Thread(target=simulator.run_client_a)
    client_a_thread.start()
    
    
    simulator.run_client_b()
    
    
    while simulator.running:
        try:
            metrics = simulator.monitor.packet_loss_queue.get_nowait()
            metrics_history.append(metrics)
        except queue.Empty:
            pass
        time.sleep(0.1)
    
    
    if metrics_history:
        plot_metrics(metrics_history)

if __name__ == "__main__":
    main()