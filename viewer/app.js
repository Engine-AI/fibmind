import * as THREE from "../node_modules/three/build/three.module.js";

const canvas = document.querySelector("#fibmind-scene");
const panel = document.querySelector("#node-panel");
const stats = document.querySelector("#stats");
const toggleMotion = document.querySelector("#toggle-motion");
const resetView = document.querySelector("#reset-view");

const layerConfig = {
  raw: { label: "原始数据层", capacity: 21, color: 0x2f80ed, z: -5.1, size: 0.26 },
  compressed: { label: "压缩数据层", capacity: 34, color: 0x13a085, z: -2.0, size: 0.34 },
  summary: { label: "摘要数据层", capacity: 55, color: 0xef9b20, z: 1.35, size: 0.45 },
  longTerm: { label: "长期记忆层", capacity: 89, color: 0xb24ac7, z: 4.65, size: 0.58 },
};

const treeSpecs = [
  { key: "code", title: "代码树", angle: -Math.PI * 0.18, radius: 5.4 },
  { key: "requirement", title: "需求树", angle: Math.PI * 0.17, radius: 5.15 },
  { key: "dialogue", title: "对话树", angle: Math.PI * 0.53, radius: 5.25 },
  { key: "error", title: "错误树", angle: Math.PI * 0.88, radius: 5.45 },
  { key: "plan", title: "计划树", angle: Math.PI * 1.22, radius: 5.2 },
  { key: "knowledge", title: "知识树", angle: Math.PI * 1.58, radius: 5.35 },
];

const crossLinks = [
  ["error-summary", "code-compressed", "错误 -> 修复代码"],
  ["requirement-summary", "plan-compressed", "需求 -> 计划"],
  ["dialogue-raw", "requirement-raw", "对话 -> 需求"],
  ["code-summary", "knowledge-longTerm", "代码模式 -> 知识"],
  ["plan-summary", "error-compressed", "计划 -> 执行反馈"],
  ["knowledge-summary", "code-longTerm", "规则 -> 代码树根"],
];

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf6f3ea);
scene.fog = new THREE.Fog(0xf6f3ea, 22, 42);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 120);
const initialCamera = new THREE.Vector3(0, -18, 11);
camera.position.copy(initialCamera);
camera.lookAt(0, 0, 0);

const root = new THREE.Group();
scene.add(root);

const ambient = new THREE.HemisphereLight(0xffffff, 0xd8c9aa, 2.2);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
keyLight.position.set(-8, -10, 14);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xffead0, 1.3);
fillLight.position.set(10, 8, 8);
scene.add(fillLight);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const nodes = new Map();
const clickable = [];
const visualLinks = [];
const adjacency = new Map();
const orbit = {
  target: new THREE.Vector3(0, 0, 0),
  radius: 21,
  theta: 0,
  phi: 1.02,
  dragging: false,
  lastX: 0,
  lastY: 0,
};
let autoRotate = true;
let selectedNode = null;

function createTextSprite(text, color = "#17211b", fontSize = 44) {
  const ratio = window.devicePixelRatio || 1;
  const canvas2d = document.createElement("canvas");
  const context = canvas2d.getContext("2d");
  context.font = `700 ${fontSize}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
  const width = Math.ceil(context.measureText(text).width + 42);
  const height = Math.ceil(fontSize + 30);
  canvas2d.width = width * ratio;
  canvas2d.height = height * ratio;
  context.scale(ratio, ratio);
  context.font = `700 ${fontSize}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
  context.fillStyle = "rgba(255, 255, 255, 0.76)";
  roundRect(context, 0, 0, width, height, 8);
  context.fill();
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, width / 2, height / 2 + 1);

  const texture = new THREE.CanvasTexture(canvas2d);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(width / 100, height / 100, 1);
  return sprite;
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

function makeMaterial(color, opacity = 1) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.58,
    metalness: 0.12,
    transparent: true,
    opacity,
  });
}

function addLine(start, end, color, opacity = 0.42, width = 2) {
  const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity, linewidth: width });
  const line = new THREE.Line(geometry, material);
  root.add(line);
  return line;
}

function addCurve(start, end, color, opacity = 0.62) {
  const mid = start.clone().add(end).multiplyScalar(0.5);
  mid.z += 2.4;
  const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
  const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(52));
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(geometry, material);
  root.add(line);
  return line;
}

function rememberLink(line, fromMesh, toMesh, kind, relation) {
  const fromId = fromMesh.userData.id;
  const toId = toMesh.userData.id;
  line.userData = {
    fromId,
    toId,
    kind,
    relation,
    baseColor: line.material.color.clone(),
    baseOpacity: line.material.opacity,
  };
  visualLinks.push(line);

  if (!adjacency.has(fromId)) adjacency.set(fromId, []);
  if (!adjacency.has(toId)) adjacency.set(toId, []);
  adjacency.get(fromId).push({ nodeId: toId, line, relation, kind });
  adjacency.get(toId).push({ nodeId: fromId, line, relation, kind });
}

function connectVisual(fromMesh, toMesh, kind, relation, color, opacity) {
  const line =
    kind === "cross"
      ? addCurve(fromMesh.position, toMesh.position, color, opacity)
      : addLine(fromMesh.position, toMesh.position, color, opacity);
  rememberLink(line, fromMesh, toMesh, kind, relation);
  return line;
}

function createLayerRing(z, radius, color) {
  const geometry = new THREE.TorusGeometry(radius, 0.015, 8, 160);
  const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.23 });
  const ring = new THREE.Mesh(geometry, material);
  ring.position.z = z;
  root.add(ring);
}

function createNode(id, spec, layerKey, index, total, role = "memory") {
  const layer = layerConfig[layerKey];
  const baseX = Math.cos(spec.angle) * spec.radius;
  const baseY = Math.sin(spec.angle) * spec.radius;
  const localRadius = 0.95 + index * 0.36;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const theta = spec.angle + index * goldenAngle;
  const x = baseX + Math.cos(theta) * localRadius;
  const y = baseY + Math.sin(theta) * localRadius;
  const z = layer.z + (index - total / 2) * 0.09;

  const geometry =
    role === "root"
      ? new THREE.IcosahedronGeometry(layer.size * 1.38, 2)
      : new THREE.SphereGeometry(layer.size, 24, 16);
  const mesh = new THREE.Mesh(geometry, makeMaterial(layer.color));
  mesh.position.set(x, y, z);
  mesh.userData = {
    id,
    title: role === "root" ? spec.title : `${spec.title} · ${layer.label}`,
    tree: spec.title,
    layer: layer.label,
    capacity: layer.capacity,
    role,
  };
  root.add(mesh);
  clickable.push(mesh);
  nodes.set(id, mesh);
  mesh.userData.baseScale = mesh.scale.x;
  mesh.userData.baseOpacity = mesh.material.opacity;
  mesh.userData.baseColor = mesh.material.color.clone();
  return mesh;
}

function buildScene() {
  Object.values(layerConfig).forEach((layer) => createLayerRing(layer.z, 9.6, layer.color));

  const axisMaterial = new THREE.LineBasicMaterial({ color: 0x6a756e, transparent: true, opacity: 0.28 });
  const axisGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, -6.6),
    new THREE.Vector3(0, 0, 6.4),
  ]);
  root.add(new THREE.Line(axisGeometry, axisMaterial));

  for (const spec of treeSpecs) {
    const rootNode = createNode(`${spec.key}-root`, spec, "longTerm", 0, 1, "root");
    const rootLabel = createTextSprite(spec.title);
    rootLabel.position.copy(rootNode.position).add(new THREE.Vector3(0, 0, 1.0));
    root.add(rootLabel);

    const layerKeys = ["raw", "compressed", "summary", "longTerm"];
    const previous = [];
    layerKeys.forEach((layerKey, layerIndex) => {
      const count = layerIndex === 0 ? 5 : layerIndex === 1 ? 4 : layerIndex === 2 ? 3 : 2;
      const current = [];
      for (let i = 0; i < count; i += 1) {
        const node = createNode(`${spec.key}-${layerKey}-${i}`, spec, layerKey, i + layerIndex, count + layerIndex);
        current.push(node);
        connectVisual(rootNode, node, "structure", "root_contains_layer_node", 0x6d7c72, 0.14);
      }
      if (previous.length) {
        current.forEach((node, i) => {
          const parent = previous[i % previous.length];
          connectVisual(node, parent, "structure", "layer_drill_down", 0x52615a, 0.28);
        });
      }
      previous.splice(0, previous.length, ...current);
    });

    nodes.set(`${spec.key}-raw`, nodes.get(`${spec.key}-raw-0`));
    nodes.set(`${spec.key}-compressed`, nodes.get(`${spec.key}-compressed-0`));
    nodes.set(`${spec.key}-summary`, nodes.get(`${spec.key}-summary-0`));
    nodes.set(`${spec.key}-longTerm`, rootNode);
  }

  crossLinks.forEach(([from, to]) => {
    const a = nodes.get(from);
    const b = nodes.get(to);
    if (a && b) connectVisual(a, b, "cross", "cross_tree_relation", 0xd94355, 0.68);
  });

  const zLabels = [
    ["Z3 长期概念", "longTerm"],
    ["Z2 摘要", "summary"],
    ["Z1 压缩", "compressed"],
    ["Z0 原始", "raw"],
  ];
  zLabels.forEach(([label, layerKey]) => {
    const sprite = createTextSprite(label, "#52615a", 34);
    sprite.position.set(-10.4, -1.2, layerConfig[layerKey].z);
    root.add(sprite);
  });
}

function updateStats() {
  stats.innerHTML = `
    <strong>${treeSpecs.length}</strong> 棵主题树<br>
    <strong>${clickable.length}</strong> 个展示节点<br>
    <strong>${crossLinks.length}</strong> 条跨树连接
  `;
}

function updateCamera() {
  const sinPhiRadius = Math.sin(orbit.phi) * orbit.radius;
  camera.position.set(
    sinPhiRadius * Math.sin(orbit.theta),
    -sinPhiRadius * Math.cos(orbit.theta),
    Math.cos(orbit.phi) * orbit.radius,
  );
  camera.lookAt(orbit.target);
}

function getReachableSubtree(anchorId) {
  const visited = new Set([anchorId]);
  const activeLines = new Set();
  const queue = [anchorId];

  while (queue.length > 0) {
    const currentId = queue.shift();
    const neighbors = adjacency.get(currentId) || [];
    neighbors.forEach((edge) => {
      activeLines.add(edge.line);
      if (!visited.has(edge.nodeId)) {
        visited.add(edge.nodeId);
        queue.push(edge.nodeId);
      }
    });
  }

  return { nodeIds: visited, lines: activeLines };
}

function applyRootedView(anchorMesh) {
  const { nodeIds, lines } = getReachableSubtree(anchorMesh.userData.id);

  clickable.forEach((node) => {
    const active = nodeIds.has(node.userData.id);
    node.material.opacity = active ? 1 : 0.12;
    node.material.color.copy(active ? node.userData.baseColor : new THREE.Color(0xaeb7b1));
    node.scale.setScalar(node === anchorMesh ? 1.72 : active ? 1.08 : 0.72);
  });

  visualLinks.forEach((line) => {
    const active = lines.has(line);
    if (active) {
      line.material.opacity = line.userData.kind === "cross" ? 0.98 : 0.82;
      line.material.color.set(line.userData.kind === "cross" ? 0xd94355 : 0x17211b);
    } else {
      line.material.opacity = 0.045;
      line.material.color.set(0xb9c0bb);
    }
  });

  return {
    nodeCount: nodeIds.size,
    lineCount: lines.size,
    crossCount: [...lines].filter((line) => line.userData.kind === "cross").length,
  };
}

function resetVisualFocus() {
  selectedNode = null;
  clickable.forEach((node) => {
    node.material.opacity = node.userData.baseOpacity;
    node.material.color.copy(node.userData.baseColor);
    node.scale.setScalar(node.userData.baseScale);
  });
  visualLinks.forEach((line) => {
    line.material.opacity = line.userData.baseOpacity;
    line.material.color.copy(line.userData.baseColor);
  });
  orbit.target.set(0, 0, 0);
}

function selectNode(mesh) {
  selectedNode = mesh;
  const focus = applyRootedView(mesh);
  const data = mesh.userData;
  orbit.target.copy(mesh.position);
  orbit.radius = Math.min(orbit.radius, 16);
  autoRotate = false;
  updateCamera();
  panel.innerHTML = `
    <span class="eyebrow">Rooted View · ${data.tree} / ${data.layer}</span>
    <h2>${data.title}</h2>
    <p>当前节点已作为搜索根。高亮范围包含 ${focus.nodeCount} 个可达节点、${focus.lineCount} 条连接线，其中 ${focus.crossCount} 条是跨树关系；灰色节点代表暂时不在这棵局部树内。</p>
  `;
}

function handlePointerMove(event) {
  if (!orbit.dragging) return;
  const dx = event.clientX - orbit.lastX;
  const dy = event.clientY - orbit.lastY;
  orbit.lastX = event.clientX;
  orbit.lastY = event.clientY;
  orbit.theta -= dx * 0.006;
  orbit.phi = Math.max(0.32, Math.min(1.38, orbit.phi + dy * 0.004));
  autoRotate = false;
  updateCamera();
}

function handleClick(event) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(clickable, false)[0];
  if (hit) selectNode(hit.object);
}

function animate() {
  requestAnimationFrame(animate);
  if (autoRotate) {
    orbit.theta += 0.0022;
    updateCamera();
  }
  root.rotation.z += 0.00035;
  renderer.render(scene, camera);
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

buildScene();
updateStats();
updateCamera();
animate();

canvas.addEventListener("pointerdown", (event) => {
  orbit.dragging = true;
  orbit.lastX = event.clientX;
  orbit.lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", handlePointerMove);
canvas.addEventListener("pointerup", (event) => {
  orbit.dragging = false;
  canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  orbit.radius = Math.max(11, Math.min(34, orbit.radius + event.deltaY * 0.012));
  updateCamera();
});
canvas.addEventListener("click", handleClick);

toggleMotion.addEventListener("click", () => {
  autoRotate = !autoRotate;
});

resetView.addEventListener("click", () => {
  resetVisualFocus();
  autoRotate = true;
  orbit.radius = 21;
  orbit.theta = 0;
  orbit.phi = 1.02;
  updateCamera();
  panel.innerHTML = `
    <span class="eyebrow">Anchor Node</span>
    <h2>任意节点可作为搜索根</h2>
    <p>拖动旋转，滚轮缩放，点击节点查看以它为根的局部记忆树，并突出相关连接线。</p>
  `;
});

window.addEventListener("resize", resize);
