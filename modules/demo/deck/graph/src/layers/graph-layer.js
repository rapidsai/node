import {CompositeLayer, COORDINATE_SYSTEM} from '@deck.gl/core';
import Attribute from '@deck.gl/core/dist/esm/lib/attribute';

import GL from '@luma.gl/constants';

import ShortestPathTransform from './shortest-path-transform';
import EdgeAttributesTransform from './edge-attributes-transform';
import NodeAttributesTransform from './node-attributes-transform';

import {ScatterplotLayer, TextLayer} from '@deck.gl/layers';
import EdgeLayer from './edge-layer';

import {TRANSITION_FRAMES, ISOCHRONIC_SCALE, ISOCHRONIC_RINGS} from './constants';

const MODE = {
  NONE: 0,
  NODE_DISTANCE: 1,
  TRAFFIC: 2,
  ISOCHRONIC: 3
};

export default class GraphLayer extends CompositeLayer {

  initializeState({gl}) {
    this.setState({
      attributes: this._getAttributes(gl),
      nodeAttributesTransform: new NodeAttributesTransform(gl),
      edgeAttributesTransform: new EdgeAttributesTransform(gl),
      shortestPathTransform: new ShortestPathTransform(gl),
      transitionDuration: 0,
      iteration: Infinity,
      lastAttributeChange: -1,
      animation: requestAnimationFrame(this.animate.bind(this))
    });
  }

  updateState({props, oldProps, changeFlags}) {
    const dataChanged = changeFlags.dataChanged || changeFlags.updateTriggersChanged;
    const {attributes, shortestPathTransform, nodeAttributesTransform, edgeAttributesTransform} = this.state;

    if (props.data && dataChanged) {
      const nodeCount = props.data.nodes.length;
      const edgeCount = props.data.edges.length;

      for (const attributeName in attributes) {
        const attribute = attributes[attributeName];

        if (changeFlags.dataChanged ||
          (changeFlags.updateTriggersChanged && changeFlags.updateTriggersChanged[attribute.userData.accessor])) {
          attribute.setNeedsUpdate();
          const isNode = attributeName.startsWith('node');
          const numInstances = isNode ? nodeCount : edgeCount;
          attribute.allocate(numInstances);
          attribute.updateBuffer({
            numInstances,
            data: isNode ? props.data.nodes : props.data.edges,
            props,
            context: this
          })
        }
      }

      // Reset model
      shortestPathTransform.update({nodeCount, edgeCount, attributes});
      nodeAttributesTransform.update({nodeCount, edgeCount, attributes});
      edgeAttributesTransform.update({nodeCount, edgeCount, attributes});

      this.setState({
        transitionDuration: edgeCount,
        maxIterations: Math.ceil(Math.sqrt(nodeCount)) + TRANSITION_FRAMES
      });
    }

    if (dataChanged || props.sourceIndex !== oldProps.sourceIndex) {
      shortestPathTransform.reset(props.sourceIndex);
      nodeAttributesTransform.reset(props.sourceIndex);
      edgeAttributesTransform.reset(props.sourceIndex);
      this.setState({iteration: 0, lastAttributeChange: 0});
    } else if (props.mode !== oldProps.mode) {
      nodeAttributesTransform.update();
      edgeAttributesTransform.update();
      if (this.state.iteration >= this.state.maxIterations) {
        this._updateAttributes();
      }
      this.setState({lastAttributeChange: this.state.iteration});
    }
  }

  finalizeState() {
    super.finalizeState();

    cancelAnimationFrame(this.state.animation);
  }

  animate() {
    if (this.state.iteration < this.state.maxIterations) {
      const {shortestPathTransform} = this.state;

      shortestPathTransform.run();

      this._updateAttributes();
    }
    this.state.iteration++;
    // Try bind the callback to the latest version of the layer
    this.state.animation = requestAnimationFrame(this.animate.bind(this));
  }

  _updateAttributes() {
    const {shortestPathTransform, nodeAttributesTransform, edgeAttributesTransform, iteration} = this.state;
    const props = this.getCurrentLayer().props;

    const moduleParameters = Object.assign(Object.create(props), {
      viewport: this.context.viewport
    });

    nodeAttributesTransform.run({
      moduleParameters,
      mode: props.mode,
      nodeValueTexture: shortestPathTransform.nodeValueTexture,
      distortion: Math.min(iteration / TRANSITION_FRAMES, 1)
    });
    edgeAttributesTransform.run({
      nodePositionsBuffer: nodeAttributesTransform.nodePositionsBuffer
    });
  }

  _getAttributes(gl) {
    return {
      nodePositions: new Attribute(gl, {
        size: 2,
        accessor: 'getNodePosition'
      }),
      nodeIndices: new Attribute(gl, {
        size: 1,
        accessor: 'getNodeIndex'
      }),
      edgeSourceIndices: new Attribute(gl, {
        size: 1,
        type: GL.INT,
        accessor: 'getEdgeSource'
      }),
      edgeTargetIndices: new Attribute(gl, {
        size: 1,
        type: GL.INT,
        accessor: 'getEdgeTarget'
      }),
      edgeValues: new Attribute(gl, {
        size: 3,
        accessor: 'getEdgeValue'
      })
    };
  }

  // Hack: we're using attribute transition with a moving target, so instead of
  // interpolating linearly within duration we make duration really long and
  // hijack the progress calculation with this easing function
  // Can probably remove when constant speed transition is implemented
  _transitionEasing(t) {
    const {iteration, lastAttributeChange} = this.state;

    const ticks = iteration - lastAttributeChange;
    if (ticks <= TRANSITION_FRAMES) {
      return ticks / TRANSITION_FRAMES;
    }
    return 1;
  }

  _getIsochronicRings() {
    const {data, getNodePosition, sourceIndex, mode} = this.props;

    const sourcePosition = getNodePosition(data.nodes[sourceIndex]);

    return mode === MODE.ISOCHRONIC && [
      new ScatterplotLayer(this.getSubLayerProps({
        id: 'isochronic-rings-circle',
        data: ISOCHRONIC_RINGS,
        filled: false,
        stroked: true,
        lineWidthMinPixels: 1,

        coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
        coordinateOrigin: sourcePosition,

        getPosition: d => [0, 0],
        getRadius: d => d * ISOCHRONIC_SCALE,
        getLineColor: [0, 128, 255]
      })),
      new TextLayer(this.getSubLayerProps({
        id: 'isochronic-rings-legend',
        data: ISOCHRONIC_RINGS,

        coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
        coordinateOrigin: sourcePosition,

        getTextAnchor: 'start',
        getPosition: d => [d * ISOCHRONIC_SCALE, 0],
        getText: d => ` ${d / 60} min`,
        getSize: 20,
        getColor: [0, 128, 255]
      }))
    ];
  }

  renderLayers() {
    const {data, getNodePosition} = this.props;
    const {nodeAttributesTransform, edgeAttributesTransform, transitionDuration} = this.state;

    const transition = this.props.transition && {
      duration: transitionDuration,
      easing: this._transitionEasing.bind(this)
    };

    return [
      new EdgeLayer(this.getSubLayerProps({
        id: 'edges',
        data: data.edges,
        getSourcePosition: d => [0, 0],
        getTargetPosition: d => [0, 0],
        getColor: [200, 200, 200],
        widthScale: 3,

        instanceSourcePositions: edgeAttributesTransform.sourcePositionsBuffer,
        instanceTargetPositions: edgeAttributesTransform.targetPositionsBuffer,
        instanceValid: edgeAttributesTransform.validityBuffer,

        transitions: transition && {
          getSourcePosition: transition,
          getTargetPosition: transition,
          getIsValid: transition
        }
      })),

      new ScatterplotLayer(this.getSubLayerProps({
        id: 'nodes',
        data: data.nodes,
        getPosition: getNodePosition,

        instancePositions: {buffer: nodeAttributesTransform.nodePositionsBuffer, size: 4},
        instanceFillColors: nodeAttributesTransform.nodeColorsBuffer,
        instanceRadius: nodeAttributesTransform.nodeRadiusBuffer,

        transitions: transition && {
          getPosition: transition,
          getFillColor: transition,
          getRadius: transition
        },

        pickable: true,
        autoHighlight: true,
        highlightColor: [0, 200, 255, 200]
      })),

      this._getIsochronicRings()
    ]
  }
}

GraphLayer.defaultProps = {
  mode: MODE.NODE_DISTANCE,
  getNodePosition: {type: 'accessor'},
  getNodeIndex: {type: 'accessor'},
  getEdgeSource: {type: 'accessor'},
  getEdgeTarget: {type: 'accessor'},
  getEdgeValue: {type: 'accessor'}
};