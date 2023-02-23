import maplibregl from 'maplibre-gl';
import React, {useEffect, useRef} from 'react';

function Map() {
  const mapContainer = useRef(null);

  useEffect(() => {
    var map = new maplibregl.Map({
      container: 'map',
      style: 'https://demotiles.maplibre.org/style.json',  // stylesheet location
      center: [-105.5, 40],                                // starting position [lng, lat]
      zoom: 3,                                             // starting zoom
      dragPan: true,
      scrollZoom: true
    });

    return () => map.remove();
  }, []);

  return <div id = 'map' ref = {mapContainer} className = 'map-container' />;
}

export default Map;
