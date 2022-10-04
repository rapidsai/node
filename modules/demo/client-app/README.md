# `react-webgl-app`

Boilerplate to create a minimal React & WebGL app.

![Example](public/images/example.png)

It is made using [rollup-react-app](https://github.com/mikbry/RollupReactApp). The WebGL code is an heavily modified example from [Mozilla](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Animating_objects_with_WebGL).

## Why
- Create a minimal from scratch React + WebGL app.
- No Three.js
- use React's hook

It will be enhanced with more examples and an article.

## How it works ?

### 1 - WebGL needs a canvas

`GLVIew` component  renders a `<canvas />` element. GLView iis using the [React Effect Hook](https://reactjs.org/docs/hooks-effect.html) to make the animation works. The code is very simple:

```javascript
const GLView = ({ width, height, scene }) => {
  const ref = useRef();

  useEffect(() => {
    const canvas = ref.current;
    const webGL = new WebGL(canvas, width, height);
    webGL.init(scene);
    return () => {
      webGL.close();
    };
  });

  return <canvas ref={ref} width={width} height={height} />;
};
```

### 2 - All the GL stuff 
`WebGL` is the engine where WebGL, shaders, model are intialized. 

The rendering animation is done using:

```javascript
    this.render = this.render.bind(this);
    this.requestId = requestAnimationFrame(this.render);
```

### 3 - Where the magic plays
`scene.js`

All the model, shaders, are here and also the scene rendering.

## Community

Don't hesitate to test, use, contribute, ...

Made by [Mik BRY](http://twitter.com/mikbry) 
