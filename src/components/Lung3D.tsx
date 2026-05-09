import { Canvas, useFrame } from "@react-three/fiber";
import { Float, OrbitControls, Sphere } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

type Status = "idle" | "loading" | "normal" | "pneumonia";

function colorForStatus(status: Status): string {
  if (status === "normal") return "#5eead4";
  if (status === "pneumonia") return "#ff5577";
  if (status === "loading") return "#a78bfa";
  return "#5fb8ff";
}

function LungLobe({
  position,
  scale,
  color,
  pulse,
}: {
  position: [number, number, number];
  scale: [number, number, number];
  color: string;
  pulse: number;
}) {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const s = 1 + Math.sin(t * pulse) * 0.04;
    ref.current.scale.set(scale[0] * s, scale[1] * s, scale[2] * s);
  });
  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshPhysicalMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.4}
        roughness={0.25}
        metalness={0.1}
        clearcoat={1}
        clearcoatRoughness={0.2}
        transmission={0.15}
        thickness={1}
      />
    </mesh>
  );
}

function Particles({ color }: { color: string }) {
  const ref = useRef<THREE.Points>(null!);
  const positions = useMemo(() => {
    const arr = new Float32Array(400 * 3);
    for (let i = 0; i < 400; i++) {
      const r = 3 + Math.random() * 2;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      arr[i * 3 + 2] = r * Math.cos(phi);
    }
    return arr;
  }, []);
  useFrame((_, dt) => {
    ref.current.rotation.y += dt * 0.05;
  });
  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color={color} size={0.04} transparent opacity={0.7} />
    </points>
  );
}

function Scene({ status }: { status: Status }) {
  const color = colorForStatus(status);
  const pulse = status === "loading" ? 6 : status === "pneumonia" ? 4 : 2;
  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[5, 5, 5]} intensity={1.2} color={color} />
      <pointLight position={[-5, -3, -5]} intensity={0.6} color="#5fb8ff" />
      <Float speed={1.2} rotationIntensity={0.6} floatIntensity={0.8}>
        <group>
          <LungLobe position={[-1.1, 0, 0]} scale={[1, 1.6, 1]} color={color} pulse={pulse} />
          <LungLobe position={[1.1, 0, 0]} scale={[1, 1.6, 1]} color={color} pulse={pulse} />
          <mesh position={[0, 0.6, 0]}>
            <cylinderGeometry args={[0.15, 0.15, 1.4, 24]} />
            <meshStandardMaterial color="#cfe7ff" emissive={color} emissiveIntensity={0.3} />
          </mesh>
          <Sphere args={[2.6, 32, 32]}>
            <meshBasicMaterial color={color} wireframe transparent opacity={0.08} />
          </Sphere>
        </group>
      </Float>
      <Particles color={color} />
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.8} />
    </>
  );
}

export function Lung3D({ status }: { status: Status }) {
  return (
    <div className="w-full h-full">
      <Canvas camera={{ position: [0, 0, 6], fov: 45 }} dpr={[1, 2]}>
        <Scene status={status} />
      </Canvas>
    </div>
  );
}
