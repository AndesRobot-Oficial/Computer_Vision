import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from interbotix_xs_msgs.srv import TorqueEnable

class ListenerControlBrazo(Node):
    def __init__(self):
        super().__init__('listener_control_brazo_rx200')

        # ---- PUBLICADORES ----
        # Usamos la misma lógica del Código 3 para el grupo
        self.pub_group = self.create_publisher(JointGroupCommand, '/rx200/commands/joint_group', 10)
        # ---- CLIENTE DE TORQUE (Para asegurar activación) ----
        self.client_torque = self.create_client(TorqueEnable, '/rx200/torque_enable')

        # ---- SUSCRIPTOR (Listener) ----
        # Escucha el array de 6 valores: [waist, shoulder, elbow, w_angle, w_rotate, gripper]
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/camera_coordinates',
            self.listener_callback,
            10
        )

        self.get_logger().info("✅ Listener RX200 (Basado en Código 3) iniciado.")
        self.activar_torque_inicial()

    def activar_torque_inicial(self):
        """Activa el torque al inicio para evitar que el brazo se caiga."""
        if self.client_torque.wait_for_service(timeout_sec=2.0):
            req = TorqueEnable.Request()
            req.cmd_type = 'all'
            req.enable = True
            self.client_torque.call_async(req)
            self.get_logger().info("⚡ Torque activado globalmente.")

    def listener_callback(self, msg):
        """Maneja los datos recibidos desde la terminal o el otro nodo."""
        try:
            if len(msg.data) >= 6:
                # 1. Extraer ángulos del brazo (Primeros 5)
                angulos_brazo = list(msg.data[:5])
                
                # 2. Extraer valor de la garra (El 6to)
                valor_garra = float(msg.data[5])

                # ---- PUBLICAR BRAZO ----
                msg_arm = JointGroupCommand()
                msg_arm.name = "arm"
                msg_arm.cmd = angulos_brazo
                self.pub_group.publish(msg_arm)

                # ---- PUBLICAR GARRA (FORMA CORRECTA) ----
                msg_grip = JointGroupCommand()
                msg_grip.name = "gripper"
                msg_grip.cmd = [float(valor_garra)]
                self.pub_group.publish(msg_grip)

                self.get_logger().info(f"🦾 Movimiento: Brazo={angulos_brazo}, Garra={valor_garra}")
            else:
                self.get_logger().warn("⚠️ Array incompleto. Se requieren 6 valores.")
        except Exception as e:
            self.get_logger().error(f"❌ Error en callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    nodo = ListenerControlBrazo()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()