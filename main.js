import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import gsap from 'gsap';

// ==============================
// CONSTANTS
// ==============================
const SPREAD = 12;
const IMG_SIZE = 256; // Smaller for faster computation on all transforms
const PYR_LEVELS = 4; // Levels 0–3

// ==============================
// THREE.JS SCENE
// ==============================
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0f172a, 0.015);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, SPREAD * 2);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

const ambientLight = new THREE.AmbientLight(0xffffff, 1);
scene.add(ambientLight);

// ── DWT planes group ──
const dwtGroup = new THREE.Group();
scene.add(dwtGroup);
const dwtPlanes = [];
const matCfg = () => new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 1 });

function createDWTPlane(w, h, x, y) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), matCfg());
    m.userData = { baseX: x, baseY: y };
    m.position.set(x, y, 0);
    dwtGroup.add(m);
    dwtPlanes.push(m);
}
createDWTPlane(SPREAD, SPREAD, 0, 0);                                                    // 0: Full
createDWTPlane(SPREAD / 2, SPREAD / 2, -SPREAD / 4, SPREAD / 4);                         // 1: LL1
createDWTPlane(SPREAD / 2, SPREAD / 2, SPREAD / 4, SPREAD / 4);                          // 2: HL1
createDWTPlane(SPREAD / 2, SPREAD / 2, -SPREAD / 4, -SPREAD / 4);                        // 3: LH1
createDWTPlane(SPREAD / 2, SPREAD / 2, SPREAD / 4, -SPREAD / 4);                         // 4: HH1
createDWTPlane(SPREAD / 4, SPREAD / 4, -SPREAD / 4 + SPREAD / 8, SPREAD / 4 + SPREAD / 8); // 5: HL2
createDWTPlane(SPREAD / 4, SPREAD / 4, -SPREAD / 4 - SPREAD / 8, SPREAD / 4 - SPREAD / 8); // 6: LH2
createDWTPlane(SPREAD / 4, SPREAD / 4, -SPREAD / 4 + SPREAD / 8, SPREAD / 4 - SPREAD / 8); // 7: HH2
// 8: Reconstruction overlay
const reconMesh = new THREE.Mesh(new THREE.PlaneGeometry(SPREAD, SPREAD), matCfg());
reconMesh.position.set(0, 0, 0);
dwtGroup.add(reconMesh);

dwtPlanes.forEach(p => p.visible = false);
reconMesh.visible = false;

// ── Pyramid planes group ──
const pyrGroup = new THREE.Group();
scene.add(pyrGroup);
const pyrPlanes = [];

for (let i = 0; i < PYR_LEVELS; i++) {
    const scale = Math.pow(0.5, i);
    const w = SPREAD * scale;
    const h = SPREAD * scale;
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), matCfg());
    m.position.set(0, 0, i * 3);
    pyrGroup.add(m);
    pyrPlanes.push(m);
}
pyrGroup.visible = false;

// ── Compression planes ──
const compGroup = new THREE.Group();
scene.add(compGroup);
const compOrigPlane = new THREE.Mesh(new THREE.PlaneGeometry(SPREAD * 0.48, SPREAD * 0.48), matCfg());
compOrigPlane.position.set(-SPREAD * 0.27, 0, 0);
compGroup.add(compOrigPlane);
const compResultPlane = new THREE.Mesh(new THREE.PlaneGeometry(SPREAD * 0.48, SPREAD * 0.48), matCfg());
compResultPlane.position.set(SPREAD * 0.27, 0, 0);
compGroup.add(compResultPlane);
compGroup.visible = false;

// ==============================
// STATE
// ==============================
let rawGray = new Float32Array(IMG_SIZE * IMG_SIZE);
let currentTab = 'wavelet';
let currentKernel = 'haar';
let dwtMode = 0;  // 0=orig, 1=lvl1, 2=lvl2, 3=recon
let explosionVal = 0;
let thresholdVal = 0;
let dwtTextures = [];
let reconTexture = null;
let pyrTextures = [];
let pyrType = 'gaussian';
let pyrDisplayMode = 'all';
let pyrHeight = 1;
let baseWaveletData = null; // full L2 transform buffer

const hiddenCanvas = document.getElementById('hidden-canvas');
const hCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

// ==============================
// WAVELET MATH
// ==============================
function haar1D(a, n) {
    const t = new Float32Array(n);
    const h = n / 2;
    for (let i = 0; i < h; i++) {
        t[i] = (a[2 * i] + a[2 * i + 1]) / Math.SQRT2;
        t[h + i] = (a[2 * i] - a[2 * i + 1]) / Math.SQRT2;
    }
    for (let i = 0; i < n; i++) a[i] = t[i];
}
function db41D(a, n) {
    const t = new Float32Array(n);
    const h = n / 2;
    const c = [0.4829629131, 0.8365163037, 0.2241438680, -0.1294095225];
    for (let i = 0; i < h; i++) {
        const j = 2 * i;
        t[i] = c[0] * a[j] + c[1] * a[(j + 1) % n] + c[2] * a[(j + 2) % n] + c[3] * a[(j + 3) % n];
        t[h + i] = c[3] * a[j] - c[2] * a[(j + 1) % n] + c[1] * a[(j + 2) % n] - c[0] * a[(j + 3) % n];
    }
    for (let i = 0; i < n; i++) a[i] = t[i];
}
function invHaar1D(a, n) {
    const t = new Float32Array(n);
    const h = n / 2;
    for (let i = 0; i < h; i++) {
        t[2 * i] = (a[i] + a[h + i]) / Math.SQRT2;
        t[2 * i + 1] = (a[i] - a[h + i]) / Math.SQRT2;
    }
    for (let i = 0; i < n; i++) a[i] = t[i];
}

function fwd1D(a, n) { (currentKernel === 'db4') ? db41D(a, n) : haar1D(a, n); }

function waveletLevel(data, size, stride) {
    for (let y = 0; y < size; y++) {
        const row = new Float32Array(size);
        for (let x = 0; x < size; x++) row[x] = data[y * stride + x];
        fwd1D(row, size);
        for (let x = 0; x < size; x++) data[y * stride + x] = row[x];
    }
    for (let x = 0; x < size; x++) {
        const col = new Float32Array(size);
        for (let y = 0; y < size; y++) col[y] = data[y * stride + x];
        fwd1D(col, size);
        for (let y = 0; y < size; y++) data[y * stride + x] = col[y];
    }
}
function invWaveletLevel(data, size, stride) {
    for (let x = 0; x < size; x++) {
        const col = new Float32Array(size);
        for (let y = 0; y < size; y++) col[y] = data[y * stride + x];
        invHaar1D(col, size);
        for (let y = 0; y < size; y++) data[y * stride + x] = col[y];
    }
    for (let y = 0; y < size; y++) {
        const row = new Float32Array(size);
        for (let x = 0; x < size; x++) row[x] = data[y * stride + x];
        invHaar1D(row, size);
        for (let x = 0; x < size; x++) data[y * stride + x] = row[x];
    }
}

// ==============================
// TEXTURE HELPERS
// ==============================
function grayToTex(data, w, h, stride, sx, sy, isDetail) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const cx = c.getContext('2d');
    const id = cx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const v = data[(sy + y) * stride + (sx + x)];
            let n;
            if (isDetail) n = Math.max(0, Math.min(255, 128 + v * 2.5));
            else          n = Math.max(0, Math.min(255, v / (currentKernel === 'db4' ? 1.5 : 2)));
            const idx = (y * w + x) * 4;
            id.data[idx] = id.data[idx + 1] = id.data[idx + 2] = n;
            id.data[idx + 3] = 255;
        }
    }
    cx.putImageData(id, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    return tex;
}
function arrayToTex(arr, w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const cx = c.getContext('2d');
    const id = cx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
        const n = Math.max(0, Math.min(255, arr[i]));
        id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = n;
        id.data[i * 4 + 3] = 255;
    }
    cx.putImageData(id, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
}

// ==============================
// IMAGE LOADING
// ==============================
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');

function loadGrayFromCanvas() {
    const d = hCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
    for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++)
        rawGray[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;
}

function generateDefaultImage() {
    hiddenCanvas.width = IMG_SIZE; hiddenCanvas.height = IMG_SIZE;
    const id = hCtx.createImageData(IMG_SIZE, IMG_SIZE);
    for (let y = 0; y < IMG_SIZE; y++) {
        for (let x = 0; x < IMG_SIZE; x++) {
            const v1 = Math.sin(x * 0.12) * Math.cos(y * 0.12) * 100;
            const v2 = (x % 32 < 16 && y % 32 < 16) ? 100 : 0;
            const v3 = (Math.sqrt((x - 128) ** 2 + (y - 128) ** 2) < 60) ? 80 : 0;
            const n = Math.max(0, Math.min(255, v1 + v2 + v3 + 50));
            const idx = (y * IMG_SIZE + x) * 4;
            id.data[idx] = id.data[idx + 1] = id.data[idx + 2] = n;
            id.data[idx + 3] = 255;
        }
    }
    hCtx.putImageData(id, 0, 0);
    loadGrayFromCanvas();
    imagePreview.src = hiddenCanvas.toDataURL();
    imagePreview.classList.add('loaded');
    onImageReady();
}

imageUpload.addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = new Image();
        img.onload = () => {
            hiddenCanvas.width = IMG_SIZE; hiddenCanvas.height = IMG_SIZE;
            hCtx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
            loadGrayFromCanvas();
            imagePreview.src = ev.target.result;
            imagePreview.classList.add('loaded');
            onImageReady();
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(f);
});

function onImageReady() {
    computeDWT();
    computePyramids();
    computeCompression();
    switchTab(currentTab);
    const loader = document.getElementById('loader');
    if (loader.classList.contains('active')) loader.classList.remove('active');
}

// ==============================
// DWT COMPUTE
// ==============================
function computeDWT() {
    dwtTextures.forEach(t => t && t.dispose());
    dwtTextures = [];
    if (reconTexture) reconTexture.dispose();
    reconTexture = null;

    // Original grayscale texture
    dwtTextures[0] = arrayToTex(rawGray, IMG_SIZE, IMG_SIZE);

    const w = new Float32Array(rawGray);
    const S = IMG_SIZE;
    const hS = S / 2;
    const qS = S / 4;

    waveletLevel(w, S, S);
    dwtTextures[1] = grayToTex(w, hS, hS, S, 0, 0, false);   // LL1
    dwtTextures[2] = grayToTex(w, hS, hS, S, hS, 0, true);   // HL1
    dwtTextures[3] = grayToTex(w, hS, hS, S, 0, hS, true);   // LH1
    dwtTextures[4] = grayToTex(w, hS, hS, S, hS, hS, true);  // HH1

    waveletLevel(w, hS, S);
    dwtTextures[5] = grayToTex(w, qS, qS, S, qS, 0, true);   // HL2
    dwtTextures[6] = grayToTex(w, qS, qS, S, 0, qS, true);   // LH2
    dwtTextures[7] = grayToTex(w, qS, qS, S, qS, qS, true);  // HH2
    dwtTextures[8] = grayToTex(w, qS, qS, S, 0, 0, false);   // LL2

    baseWaveletData = new Float32Array(w);

    drawEnergyChart(w);
    setDWTMode(dwtMode);
}

// ── Inverse DWT with thresholding ──
function reconstructFromThreshold() {
    if (!baseWaveletData) return;
    const w = new Float32Array(baseWaveletData);
    const S = IMG_SIZE;
    const hS = S / 2;
    const qS = S / 4;
    let totalCoeffs = 0;
    let zeroedCoeffs = 0;

    // Threshold detail coefficients only (not LL2 approximation)
    function thresh(sx, sy, sz) {
        for (let y = 0; y < sz; y++) {
            for (let x = 0; x < sz; x++) {
                totalCoeffs++;
                if (Math.abs(w[(sy + y) * S + (sx + x)]) < thresholdVal) {
                    w[(sy + y) * S + (sx + x)] = 0;
                    zeroedCoeffs++;
                }
            }
        }
    }
    // L2 details
    thresh(qS, 0, qS); thresh(0, qS, qS); thresh(qS, qS, qS);
    // L1 details
    thresh(hS, 0, hS); thresh(0, hS, hS); thresh(hS, hS, hS);

    // Inverse
    invWaveletLevel(w, hS, S);
    invWaveletLevel(w, S, S);

    // Flatten to 1D for PSNR
    const recon = new Float32Array(S * S);
    for (let i = 0; i < S * S; i++) recon[i] = w[i];

    // PSNR
    let mse = 0;
    for (let i = 0; i < S * S; i++) mse += (rawGray[i] - recon[i]) ** 2;
    mse /= (S * S);
    const psnr = mse > 0 ? (10 * Math.log10(255 * 255 / mse)).toFixed(2) : '∞';

    document.getElementById('metric-psnr').textContent = psnr + ' dB';
    document.getElementById('metric-zeroed').textContent = ((zeroedCoeffs / totalCoeffs) * 100).toFixed(1) + '%';
    document.getElementById('recon-metrics').style.display = 'flex';

    if (reconTexture) reconTexture.dispose();
    reconTexture = arrayToTex(recon, S, S);

    // Show reconstruction
    dwtMode = 3;
    dwtPlanes.forEach(p => p.visible = false);
    reconMesh.visible = true;
    reconMesh.material.map = reconTexture;
    reconMesh.material.needsUpdate = true;
}

// ── Energy Chart ──
function drawEnergyChart(wData) {
    const canvas = document.getElementById('energy-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const S = IMG_SIZE;
    const hS = S / 2;
    const qS = S / 4;

    function bandEnergy(sx, sy, sz) {
        let e = 0;
        for (let y = 0; y < sz; y++)
            for (let x = 0; x < sz; x++)
                e += wData[(sy + y) * S + (sx + x)] ** 2;
        return e;
    }
    const bands = [
        { name: 'LL2', e: bandEnergy(0, 0, qS) },
        { name: 'HL2', e: bandEnergy(qS, 0, qS) },
        { name: 'LH2', e: bandEnergy(0, qS, qS) },
        { name: 'HH2', e: bandEnergy(qS, qS, qS) },
        { name: 'HL1', e: bandEnergy(hS, 0, hS) },
        { name: 'LH1', e: bandEnergy(0, hS, hS) },
        { name: 'HH1', e: bandEnergy(hS, hS, hS) },
    ];
    const totalE = bands.reduce((s, b) => s + b.e, 0);
    bands.forEach(b => b.pct = totalE > 0 ? b.e / totalE : 0);

    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const barW = (W - 20) / bands.length;
    const maxPct = Math.max(...bands.map(b => b.pct), 0.01);

    bands.forEach((b, i) => {
        const bH = (b.pct / maxPct) * (H - 30);
        const x = 10 + i * barW;
        const y = H - 15 - bH;

        const grad = ctx.createLinearGradient(x, y, x, H - 15);
        grad.addColorStop(0, i === 0 ? '#10b981' : '#8b5cf6');
        grad.addColorStop(1, i === 0 ? '#064e3b' : '#312e81');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x + 2, y, barW - 4, bH, 3);
        ctx.fill();

        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(b.name, x + barW / 2, H - 3);
        ctx.fillStyle = '#c4b5fd';
        ctx.font = '8px JetBrains Mono';
        ctx.fillText((b.pct * 100).toFixed(1) + '%', x + barW / 2, y - 3);
    });
}

// ==============================
// PYRAMID COMPUTE (Ch 7)
// ==============================
function computePyramids() {
    pyrTextures.forEach(t => t && t.dispose());
    pyrTextures = [];

    // Gaussian Pyramid: each level is low‑pass filtered then downsampled
    const gaussLevels = [new Float32Array(rawGray)];
    let curSize = IMG_SIZE;

    for (let lvl = 1; lvl < PYR_LEVELS; lvl++) {
        const prev = gaussLevels[lvl - 1];
        const newSize = curSize / 2;
        const next = new Float32Array(newSize * newSize);

        // 5×5 Gaussian kernel approximation via simple box average of 2×2 with overlap
        for (let y = 0; y < newSize; y++) {
            for (let x = 0; x < newSize; x++) {
                let sum = 0, count = 0;
                for (let dy = -1; dy <= 2; dy++) {
                    for (let dx = -1; dx <= 2; dx++) {
                        const sy = Math.max(0, Math.min(curSize - 1, y * 2 + dy));
                        const sx = Math.max(0, Math.min(curSize - 1, x * 2 + dx));
                        // Gaussian-like weights: center heavier
                        const w = (dy >= 0 && dy <= 1 && dx >= 0 && dx <= 1) ? 4 : 1;
                        sum += prev[sy * curSize + sx] * w;
                        count += w;
                    }
                }
                next[y * newSize + x] = sum / count;
            }
        }
        gaussLevels.push(next);
        curSize = newSize;
    }

    if (pyrType === 'gaussian') {
        let sz = IMG_SIZE;
        for (let i = 0; i < PYR_LEVELS; i++) {
            pyrTextures[i] = arrayToTex(gaussLevels[i], sz, sz);
            sz /= 2;
        }
    } else {
        // Laplacian: L_k = G_k - expand(G_{k+1})
        let sz = IMG_SIZE;
        for (let i = 0; i < PYR_LEVELS - 1; i++) {
            const curG = gaussLevels[i];
            const nextG = gaussLevels[i + 1];
            const nextSz = sz / 2;
            // Expand next level
            const expanded = new Float32Array(sz * sz);
            for (let y = 0; y < sz; y++) {
                for (let x = 0; x < sz; x++) {
                    const nx = Math.min(nextSz - 1, Math.floor(x / 2));
                    const ny = Math.min(nextSz - 1, Math.floor(y / 2));
                    expanded[y * sz + x] = nextG[ny * nextSz + nx];
                }
            }
            const lap = new Float32Array(sz * sz);
            for (let i2 = 0; i2 < sz * sz; i2++) lap[i2] = 128 + (curG[i2] - expanded[i2]) * 2;
            pyrTextures[i] = arrayToTex(lap, sz, sz);
            sz /= 2;
        }
        // Last level = residual (just the smallest Gauss)
        const lastSz = IMG_SIZE / Math.pow(2, PYR_LEVELS - 1);
        pyrTextures[PYR_LEVELS - 1] = arrayToTex(gaussLevels[PYR_LEVELS - 1], lastSz, lastSz);
    }

    setPyramidMode();
}

// ==============================
// COMPRESSION COMPUTE (Ch 8)
// ==============================
let huffmanResult = null;
let rleResult = null;

function computeCompression() {
    computeHuffman();
    computeRLE(128);
    updateCompressionDisplay();
}

// ── Huffman ──
function computeHuffman() {
    const hist = new Array(256).fill(0);
    for (let i = 0; i < rawGray.length; i++) hist[Math.round(rawGray[i])]++;
    const total = rawGray.length;

    // Build Huffman tree
    let nodes = [];
    for (let i = 0; i < 256; i++) {
        if (hist[i] > 0) nodes.push({ sym: i, freq: hist[i], left: null, right: null });
    }
    while (nodes.length > 1) {
        nodes.sort((a, b) => a.freq - b.freq);
        const l = nodes.shift(), r = nodes.shift();
        nodes.push({ sym: -1, freq: l.freq + r.freq, left: l, right: r });
    }
    const root = nodes[0];

    // Generate codes
    const codes = {};
    function traverse(node, prefix) {
        if (!node) return;
        if (node.sym >= 0) { codes[node.sym] = prefix || '0'; return; }
        traverse(node.left, prefix + '0');
        traverse(node.right, prefix + '1');
    }
    traverse(root, '');

    // Calculate sizes
    let compBits = 0;
    for (let sym in codes) compBits += hist[parseInt(sym)] * codes[sym].length;
    const origBits = total * 8;

    // Entropy
    let entropy = 0;
    for (let i = 0; i < 256; i++) {
        if (hist[i] > 0) {
            const p = hist[i] / total;
            entropy -= p * Math.log2(p);
        }
    }

    huffmanResult = { hist, codes, origBits, compBits, entropy, total };

    // Draw histogram
    drawHistogram(hist);

    // Populate table
    const table = document.getElementById('huffman-table');
    table.innerHTML = '';
    const sorted = Object.entries(codes).sort((a, b) => hist[parseInt(b[0])] - hist[parseInt(a[0])]);
    sorted.slice(0, 20).forEach(([sym, code]) => {
        const row = document.createElement('div');
        row.className = 'code-row';
        row.innerHTML = `<span class="sym">${sym}</span><span class="freq">${hist[parseInt(sym)]}</span><span class="bits">${code}</span>`;
        table.appendChild(row);
    });

    document.getElementById('huff-orig').textContent = (origBits / 8 / 1024).toFixed(1) + ' KB';
    document.getElementById('huff-comp').textContent = (compBits / 8 / 1024).toFixed(1) + ' KB';
    document.getElementById('huff-ratio').textContent = (origBits / compBits).toFixed(2) + ':1';
    document.getElementById('huff-entropy').textContent = entropy.toFixed(3) + ' bits';
}

function drawHistogram(hist) {
    const canvas = document.getElementById('histogram-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const maxH = Math.max(...hist);
    const barW = W / 256;
    for (let i = 0; i < 256; i++) {
        const bH = (hist[i] / maxH) * (H - 6);
        const grad = ctx.createLinearGradient(0, H - bH, 0, H);
        grad.addColorStop(0, '#8b5cf6');
        grad.addColorStop(1, '#312e81');
        ctx.fillStyle = grad;
        ctx.fillRect(i * barW, H - bH, Math.max(barW - 0.3, 0.5), bH);
    }
}

// ── Run-Length Encoding ──
function computeRLE(threshold) {
    const total = IMG_SIZE * IMG_SIZE;
    let runs = 0;
    let prevBit = (rawGray[0] >= threshold) ? 1 : 0;
    runs = 1;
    for (let i = 1; i < total; i++) {
        const bit = (rawGray[i] >= threshold) ? 1 : 0;
        if (bit !== prevBit) { runs++; prevBit = bit; }
    }
    rleResult = { total, runs, threshold };

    document.getElementById('rle-orig').textContent = total.toLocaleString();
    document.getElementById('rle-runs').textContent = runs.toLocaleString();
    document.getElementById('rle-ratio').textContent = (total / runs).toFixed(2) + ':1';

    // Generate binary image texture for display
    const bin = new Float32Array(total);
    for (let i = 0; i < total; i++) bin[i] = (rawGray[i] >= threshold) ? 255 : 0;
    const tex = arrayToTex(bin, IMG_SIZE, IMG_SIZE);
    compResultPlane.material.map = tex;
    compResultPlane.material.needsUpdate = true;
}

// ── Wavelet Lossy Compression (JPEG2000 concept) ──
function waveletCompress(thresh) {
    const w = new Float32Array(rawGray);
    const S = IMG_SIZE;
    const hS = S / 2;
    const qS = S / 4;

    // Forward 2‑level DWT
    waveletLevel(w, S, S);
    waveletLevel(w, hS, S);

    let totalC = 0, zeroedC = 0;
    function applyT(sx, sy, sz) {
        for (let y = 0; y < sz; y++) {
            for (let x = 0; x < sz; x++) {
                totalC++;
                if (Math.abs(w[(sy + y) * S + (sx + x)]) < thresh) {
                    w[(sy + y) * S + (sx + x)] = 0;
                    zeroedC++;
                }
            }
        }
    }
    applyT(qS, 0, qS); applyT(0, qS, qS); applyT(qS, qS, qS);
    applyT(hS, 0, hS); applyT(0, hS, hS); applyT(hS, hS, hS);

    // Inverse
    invWaveletLevel(w, hS, S);
    invWaveletLevel(w, S, S);

    let mse = 0;
    for (let i = 0; i < S * S; i++) mse += (rawGray[i] - w[i]) ** 2;
    mse /= (S * S);
    const psnr = mse > 0 ? (10 * Math.log10(255 * 255 / mse)).toFixed(2) : '∞';

    document.getElementById('wc-psnr').textContent = psnr + ' dB';
    document.getElementById('wc-zeroed').textContent = ((zeroedC / totalC) * 100).toFixed(1) + '%';
    document.getElementById('wc-ratio').textContent = totalC > 0 ? (totalC / (totalC - zeroedC)).toFixed(2) + ':1' : '—';

    const tex = arrayToTex(w, S, S);
    compResultPlane.material.map = tex;
    compResultPlane.material.needsUpdate = true;
}

function updateCompressionDisplay() {
    compOrigPlane.material.map = dwtTextures[0];
    compOrigPlane.material.needsUpdate = true;
}

// ==============================
// DWT DISPLAY MODES
// ==============================
function setDWTMode(lvl) {
    dwtMode = lvl;
    dwtPlanes.forEach(p => p.visible = false);
    reconMesh.visible = false;

    if (lvl === 0) {
        dwtPlanes[0].visible = true;
        dwtPlanes[0].material.map = dwtTextures[0];
    } else if (lvl === 1) {
        for (let i = 1; i <= 4; i++) {
            dwtPlanes[i].visible = true;
            dwtPlanes[i].material.map = dwtTextures[i];
        }
        dwtPlanes[1].scale.set(1, 1, 1);
        dwtPlanes[1].position.set(dwtPlanes[1].userData.baseX, dwtPlanes[1].userData.baseY, 0);
    } else if (lvl === 2) {
        [2, 3, 4, 5, 6, 7].forEach(i => {
            dwtPlanes[i].visible = true;
            dwtPlanes[i].material.map = dwtTextures[i];
        });
        dwtPlanes[1].visible = true;
        dwtPlanes[1].material.map = dwtTextures[8];
        dwtPlanes[1].scale.set(0.5, 0.5, 0.5);
        dwtPlanes[1].position.set(-SPREAD / 4 - SPREAD / 8, SPREAD / 4 + SPREAD / 8, 0);
        dwtPlanes[1].userData.baseX2 = dwtPlanes[1].position.x;
        dwtPlanes[1].userData.baseY2 = dwtPlanes[1].position.y;
    } else if (lvl === 3) {
        reconMesh.visible = true;
    }
    applyExplosion();
}

function applyExplosion() {
    dwtPlanes.forEach((p, i) => {
        if (!p.visible || i === 0) return;
        let bx = p.userData.baseX, by = p.userData.baseY;
        if (dwtMode === 2 && i === 1) { bx = p.userData.baseX2; by = p.userData.baseY2; }
        gsap.to(p.position, {
            x: bx + Math.sign(bx) * explosionVal,
            y: by + Math.sign(by) * explosionVal,
            duration: 0.8, ease: 'back.out(1.2)'
        });
    });
}

// ==============================
// PYRAMID DISPLAY
// ==============================
function setPyramidMode() {
    pyrPlanes.forEach((p, i) => {
        if (pyrTextures[i]) {
            p.material.map = pyrTextures[i];
            p.material.needsUpdate = true;
        }
    });

    if (pyrDisplayMode === 'all') {
        pyrPlanes.forEach((p, i) => {
            p.visible = true;
            gsap.to(p.position, { z: i * 3 * pyrHeight, duration: 0.6, ease: 'power2.out' });
        });
    } else {
        const idx = parseInt(pyrDisplayMode);
        pyrPlanes.forEach((p, i) => {
            p.visible = (i === idx);
            p.position.z = 0;
        });
    }
}

// ==============================
// TAB SWITCHING
// ==============================
function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    const el = document.getElementById('tab-' + tab);
    if (el) el.classList.add('active');

    dwtGroup.visible = (tab === 'wavelet');
    pyrGroup.visible = (tab === 'pyramid');
    compGroup.visible = (tab === 'compress');

    if (tab === 'wavelet') setDWTMode(dwtMode);
    if (tab === 'pyramid') setPyramidMode();
    if (tab === 'compress') updateCompressionDisplay();
}

// ==============================
// UI WIRING
// ==============================
// Tab buttons
document.querySelectorAll('.tab-btn').forEach(b => {
    b.addEventListener('click', () => switchTab(b.dataset.tab));
});

// DWT controls
const b0 = document.getElementById('btn-orig');
const b1 = document.getElementById('btn-lvl1');
const b2 = document.getElementById('btn-lvl2');
const bHaar = document.getElementById('btn-haar');
const bDb4 = document.getElementById('btn-db4');
const dwtBtns = [b0, b1, b2];
const kernBtns = [bHaar, bDb4];
function act(btn, grp) { grp.forEach(b => b.classList.remove('active')); btn.classList.add('active'); }

b0.addEventListener('click', () => { act(b0, dwtBtns); setDWTMode(0); });
b1.addEventListener('click', () => { act(b1, dwtBtns); setDWTMode(1); });
b2.addEventListener('click', () => { act(b2, dwtBtns); setDWTMode(2); });
bHaar.addEventListener('click', () => { act(bHaar, kernBtns); currentKernel = 'haar'; computeDWT(); });
bDb4.addEventListener('click', () => { act(bDb4, kernBtns); currentKernel = 'db4'; computeDWT(); });

document.getElementById('explodeSlider').addEventListener('input', (e) => {
    explosionVal = parseFloat(e.target.value) / 20;
    document.getElementById('explode-val').textContent = e.target.value + '%';
    applyExplosion();
});
document.getElementById('threshSlider').addEventListener('input', (e) => {
    thresholdVal = parseFloat(e.target.value);
    document.getElementById('thresh-val').textContent = thresholdVal;
});
document.getElementById('btn-recon').addEventListener('click', () => {
    act(document.getElementById('btn-recon'), []);
    reconstructFromThreshold();
});

// Pyramid controls
const bGauss = document.getElementById('btn-gauss');
const bLapl = document.getElementById('btn-laplace');
const pyrBtns = [bGauss, bLapl];
bGauss.addEventListener('click', () => { act(bGauss, pyrBtns); pyrType = 'gaussian'; computePyramids(); });
bLapl.addEventListener('click', () => { act(bLapl, pyrBtns); pyrType = 'laplacian'; computePyramids(); });

const pyrLvlBtns = ['btn-pyr-all', 'btn-pyr-0', 'btn-pyr-1', 'btn-pyr-2', 'btn-pyr-3'].map(id => document.getElementById(id));
function setPyrDisplay(mode) {
    pyrDisplayMode = mode;
    setPyramidMode();
}
document.getElementById('btn-pyr-all').addEventListener('click', () => { act(pyrLvlBtns[0], pyrLvlBtns); setPyrDisplay('all'); });
document.getElementById('btn-pyr-0').addEventListener('click', () => { act(pyrLvlBtns[1], pyrLvlBtns); setPyrDisplay('0'); });
document.getElementById('btn-pyr-1').addEventListener('click', () => { act(pyrLvlBtns[2], pyrLvlBtns); setPyrDisplay('1'); });
document.getElementById('btn-pyr-2').addEventListener('click', () => { act(pyrLvlBtns[3], pyrLvlBtns); setPyrDisplay('2'); });
document.getElementById('btn-pyr-3').addEventListener('click', () => { act(pyrLvlBtns[4], pyrLvlBtns); setPyrDisplay('3'); });

document.getElementById('pyrHeightSlider').addEventListener('input', (e) => {
    pyrHeight = parseFloat(e.target.value) / 100;
    document.getElementById('pyr-height-val').textContent = e.target.value + '%';
    setPyramidMode();
});

// Compression controls
const compBtns = ['btn-huffman', 'btn-rle', 'btn-wavelet-compress'].map(id => document.getElementById(id));
const compSections = ['huffman-section', 'rle-section', 'wavelet-compress-section'].map(id => document.getElementById(id));

function showCompSection(idx) {
    compSections.forEach((s, i) => s.style.display = i === idx ? 'flex' : 'none');
    act(compBtns[idx], compBtns);
}
compBtns[0].addEventListener('click', () => showCompSection(0));
compBtns[1].addEventListener('click', () => showCompSection(1));
compBtns[2].addEventListener('click', () => showCompSection(2));

document.getElementById('rleThreshSlider').addEventListener('input', (e) => {
    const v = parseInt(e.target.value);
    document.getElementById('rle-thresh-val').textContent = v;
    computeRLE(v);
});

document.getElementById('wcThreshSlider').addEventListener('input', (e) => {
    document.getElementById('wc-thresh-val').textContent = e.target.value;
});
document.getElementById('btn-wc-apply').addEventListener('click', () => {
    waveletCompress(parseFloat(document.getElementById('wcThreshSlider').value));
});

// ==============================
// RENDER LOOP
// ==============================
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    const t = Date.now() * 0.001;
    const activeGroup = currentTab === 'wavelet' ? dwtGroup : currentTab === 'pyramid' ? pyrGroup : compGroup;
    activeGroup.position.y = Math.sin(t) * 0.15;
    activeGroup.rotation.x = Math.sin(t * 0.5) * 0.03;
    activeGroup.rotation.y = Math.cos(t * 0.35) * 0.03;

    renderer.render(scene, camera);
}

generateDefaultImage();
animate();
