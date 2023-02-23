import maplibregl from 'maplibre-gl';
import React, {useContext, useEffect, useRef} from 'react';

function Map({props, updateTransform}) {
  const mapContainer = useRef(null);

  useEffect(() => {
    var map = new maplibregl.Map({
      container: 'map',
      style: 'https://demotiles.maplibre.org/style.json',  // stylesheet location
      center: [-105, 40],                                  // starting position [lng, lat]
      zoom: 5,                                             // starting zoom
      dragPan: true,
      scrollZoom: true
    });
    map.on('move', function(e) { updateTransform(e.target.transform); });

    return () => map.remove();
  }, []);

  return <div id = 'map' ref = {mapContainer} className = 'map-container' />;
}

export default Map;
