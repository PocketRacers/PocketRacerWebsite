(() => {
  const viewerElements = Array.from(document.querySelectorAll(".stl-viewer[data-src]"));
  if (!viewerElements.length) return;

  const setFallback = (element, message) => {
    let fallback = element.querySelector(".stl-fallback");
    if (!fallback) {
      fallback = document.createElement("div");
      fallback.className = "stl-fallback";
      element.appendChild(fallback);
    }
    fallback.textContent = message;
  };

  if (!window.THREE || !THREE.WebGLRenderer) {
    viewerElements.forEach((el) => setFallback(el, "3D preview unavailable (Three.js not loaded)."));
    return;
  }
  if (!THREE.STLLoader) {
    viewerElements.forEach((el) => setFallback(el, "3D preview unavailable (STL loader missing)."));
    return;
  }
  if (!THREE.OrbitControls) {
    viewerElements.forEach((el) => setFallback(el, "3D preview unavailable (controls missing)."));
    return;
  }

  const instancesByElement = new WeakMap();
  const instances = [];

  const intersectionObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const instance = instancesByElement.get(entry.target);
        if (!instance) continue;
        instance.visible = entry.isIntersecting;
      }
    },
    { rootMargin: "200px 0px" },
  );

  const resizeOne = (instance, width, height) => {
    const safeWidth = Math.max(1, Math.floor(width));
    const safeHeight = Math.max(1, Math.floor(height));
    instance.camera.aspect = safeWidth / safeHeight;
    instance.camera.updateProjectionMatrix();
    instance.renderer.setSize(safeWidth, safeHeight, false);
  };

  const resizeObserver =
    typeof ResizeObserver !== "undefined"
      ? new ResizeObserver((entries) => {
          for (const entry of entries) {
            const instance = instancesByElement.get(entry.target);
            if (!instance) continue;
            resizeOne(instance, entry.contentRect.width, entry.contentRect.height);
          }
        })
      : null;

  const initViewer = (element) => {
    setFallback(element, "Loading 3D preview…");

    const width = element.clientWidth || 320;
    const height = element.clientHeight || 220;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    if ("outputColorSpace" in renderer && THREE.SRGBColorSpace) {
      renderer.outputColorSpace = THREE.SRGBColorSpace;
    } else if ("outputEncoding" in renderer && THREE.sRGBEncoding) {
      renderer.outputEncoding = THREE.sRGBEncoding;
    }

    element.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    camera.position.set(0, 0, 120);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.2;
    controls.minDistance = 20;
    controls.maxDistance = 500;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.1));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(3, 6, 4);
    scene.add(dirLight);

    const instance = { element, renderer, scene, camera, controls, visible: false };
    instances.push(instance);
    instancesByElement.set(element, instance);
    intersectionObserver.observe(element);
    resizeObserver?.observe(element);

    const loader = new THREE.STLLoader();
    const src = encodeURI(element.dataset.src || "");
    loader.load(
      src,
      (geometry) => {
        if (geometry.computeVertexNormals) geometry.computeVertexNormals();

        const color = element.dataset.color || "#9aa0a6";
        const material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(color),
          metalness: 0.05,
          roughness: 0.85,
        });
        const mesh = new THREE.Mesh(geometry, material);

        const box = new THREE.Box3().setFromObject(mesh);
        const size = new THREE.Vector3();
        const center = new THREE.Vector3();
        box.getSize(size);
        box.getCenter(center);

        mesh.position.sub(center);
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const scale = 70 / maxDim;
        mesh.scale.setScalar(scale);

        scene.add(mesh);
        controls.target.set(0, 0, 0);
        controls.update();

        const fallback = element.querySelector(".stl-fallback");
        fallback?.remove();
      },
      undefined,
      () => setFallback(element, "Preview failed to load. Use the Download link."),
    );
  };

  for (const element of viewerElements) initViewer(element);

  if (!resizeObserver) {
    window.addEventListener("resize", () => {
      for (const instance of instances) {
        const rect = instance.element.getBoundingClientRect();
        resizeOne(instance, rect.width, rect.height);
      }
    });
  }

  const renderLoop = () => {
    requestAnimationFrame(renderLoop);
    for (const instance of instances) {
      if (!instance.visible) continue;
      instance.controls.update();
      instance.renderer.render(instance.scene, instance.camera);
    }
  };
  renderLoop();
})();

