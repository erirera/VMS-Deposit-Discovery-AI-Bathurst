// Map Configuration & Initialization
const mapConfig = {
    center: [47.5, -66.0], // Approx center of Bathurst Mining Camp, NB
    zoom: 9,
    minZoom: 6,
    maxZoom: 14
};

const map = L.map('map', {
    center: mapConfig.center,
    zoom: mapConfig.zoom,
    minZoom: mapConfig.minZoom,
    maxZoom: mapConfig.maxZoom,
    zoomControl: false // Custom placement later
});

// Move zoom control to top-right
L.control.zoom({
    position: 'topright'
}).addTo(map);

// Add Dark Matter Basemap (CartoDB) for a premium dark theme look
const basemap = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// --- Mock Data Generation --- //

// 1. BMC Boundary Polygon
const bmcBoundsData = {
    type: "Feature",
    properties: { name: "Bathurst Mining Camp", type: "Boundary" },
    geometry: {
        type: "Polygon",
        coordinates: [[
            [-66.8, 47.2],
            [-65.3, 47.2],
            [-65.3, 47.9],
            [-66.8, 47.9],
            [-66.8, 47.2]
        ]]
    }
};

// 2. Generate random points for Deposits (Positive)
const depositsData = {
    type: "FeatureCollection",
    features: []
};
const barrenData = {
    type: "FeatureCollection",
    features: []
};
const heatMapPoints = []; // [lat, lng, intensity]

function generateRandomPoints(count, isDeposit) {
    for (let i = 0; i < count; i++) {
        // Randomly distribute within BMC bounds roughly
        const lat = 47.25 + Math.random() * 0.6;
        const lng = -66.7 + Math.random() * 1.3;
        
        const feature = {
            type: "Feature",
            properties: { 
                id: `Hole-${Math.floor(Math.random()*10000)}`,
                status: isDeposit ? 'Positive VMS' : 'Barren',
                depth: Math.floor(Math.random() * 800) + 50
            },
            geometry: {
                type: "Point",
                coordinates: [lng, lat] // GeoJSON is [lng, lat]
            }
        };

        if (isDeposit) {
            depositsData.features.push(feature);
            // High intensity for deposits
            heatMapPoints.push([lat, lng, 0.9]);
            // Add some clustered 'hidden' high probability targets around deposits
            if(Math.random() > 0.5) {
                heatMapPoints.push([lat + (Math.random()-0.5)*0.05, lng + (Math.random()-0.5)*0.05, 0.7 + Math.random()*0.3]);
            }
        } else {
            barrenData.features.push(feature);
            // Low intensity near barren holes
            if(Math.random() > 0.8) heatMapPoints.push([lat, lng, 0.1]); 
        }
    }
}

// Generate 45 deposits and 250 barren holes (reflecting real-world class imbalance)
generateRandomPoints(45, true);
generateRandomPoints(250, false);

// Add some random high probability "undiscovered" zones
for(let i=0; i<30; i++) {
    heatMapPoints.push([47.3 + Math.random() * 0.5, -66.6 + Math.random() * 1.1, 0.6 + Math.random()*0.4]);
}

// Update UI stat counter
document.getElementById('target-count').innerText = "30+ Priority";

// --- Leaflet Layer Definitions --- //

// Custom Icons
const createCustomIcon = (type) => {
    return L.divIcon({
        className: type === 'deposit' ? 'custom-deposit-marker' : 'custom-barren-marker',
        iconSize: type === 'deposit' ? [14, 14] : [8, 8],
        iconAnchor: type === 'deposit' ? [7, 7] : [4, 4]
    });
};

// 1. Boundary Layer
const bmcLayer = L.geoJSON(bmcBoundsData, {
    style: {
        color: 'var(--accent-blue)',
        weight: 2,
        dashArray: '5, 10',
        fillColor: 'var(--accent-blue)',
        fillOpacity: 0.05
    }
});

// Tooltip helper
const onEachFeature = (feature, layer) => {
    if (feature.properties) {
        let tpContent = `<strong>${feature.properties.id || feature.properties.name}</strong><br>`;
        if(feature.properties.status) tpContent += `Status: <span style="color:${feature.properties.status === 'Barren' ? '#94a3b8' : '#f59e0b'}">${feature.properties.status}</span><br>`;
        if(feature.properties.depth) tpContent += `Depth: ${feature.properties.depth}m`;
        
        layer.bindTooltip(tpContent, {
            className: 'glass-tooltip',
            direction: 'top',
            offset: [0, -10]
        });
    }
};

// 2. Deposits Layer
const depositsLayer = L.geoJSON(depositsData, {
    pointToLayer: (feature, latlng) => {
        return L.marker(latlng, {icon: createCustomIcon('deposit')});
    },
    onEachFeature: onEachFeature
});

// 3. Barren Layer
const barrenLayer = L.geoJSON(barrenData, {
    pointToLayer: (feature, latlng) => {
        return L.marker(latlng, {icon: createCustomIcon('barren')});
    },
    onEachFeature: onEachFeature
});

// 4. Heatmap Layer
// Using leaflet-heat plugin
const heatLayer = L.heatLayer(heatMapPoints, {
    radius: 35,
    blur: 25,
    maxZoom: 12,
    gradient: {
        0.2: 'rgba(59,130,246,0.3)', // blue
        0.5: 'rgba(16,185,129,0.7)', // green
        0.8: 'rgba(245,158,11,0.9)', // gold
        1.0: 'rgba(239,68,68,1)'     // red - high prob
    }
});


// Add initial layers to map based on checkboxes
bmcLayer.addTo(map);
depositsLayer.addTo(map);
barrenLayer.addTo(map);

// --- Interactions --- //

// Checkbox Logic
document.getElementById('layer-bmc').addEventListener('change', function(e) {
    if(this.checked) map.addLayer(bmcLayer);
    else map.removeLayer(bmcLayer);
});

document.getElementById('layer-deposits').addEventListener('change', function(e) {
    if(this.checked) map.addLayer(depositsLayer);
    else map.removeLayer(depositsLayer);
});

document.getElementById('layer-barren').addEventListener('change', function(e) {
    if(this.checked) map.addLayer(barrenLayer);
    else map.removeLayer(barrenLayer);
});

document.getElementById('layer-heatmap').addEventListener('change', function(e) {
    if(this.checked) {
        map.addLayer(heatLayer);
        // Bring deposits to front so they aren't hidden by heat
        if(map.hasLayer(depositsLayer)) {
            depositsLayer.bringToFront();
        }
    }
    else map.removeLayer(heatLayer);
});

// GSAP Animations for Sidebar
gsap.from(".sidebar", {
    x: -420,
    duration: 1,
    ease: "power3.out"
});

gsap.from(".info-section, .control-section, .stats-card", {
    opacity: 0,
    y: 20,
    stagger: 0.15,
    duration: 0.8,
    delay: 0.3,
    ease: "power2.out"
});
