function generateLegend(container, keys) {
  // select the svg area
  var svg = d3.select(container)

  // Add one dot in the legend for each name.
  var size = 20
  svg.selectAll('mydots')
    .data(Object.keys(keys))
    .enter()
    .append('rect')
    .attr('x', 10)
    .attr('y',
          function(d, i) { return 10 + i * (size + 5) })  // 100 is where the first dot appears.
                                                          // 25 is the distance between dots
    .attr('width', size)
    .attr('height', size)
    .style('fill',
           function(d) {
             values = thresholdScale(d);
             return 'rgb(' + values.join(', ') + ')';
           })
    .style('opacity', 0.8)

  // Add one dot in the legend for each name.
  svg.selectAll('mylabels')
    .data(Object.keys(keys))
    .enter()
    .append('text')
    .attr('x', 10 + size * 1.2)
    .attr('y',
          function(d, i) {
            return 10 + i * (size + 5) + (size / 2)
          })  // 100 is where the first dot appears. 25 is the distance between dots
    .style('fill',
           function(d) {
             values = thresholdScale(d);
             console.log(d, values);
             return 'rgb(' + values.join(', ') + ')';
           })
    .text(function(d) { return keys[d] })
    .attr('text-anchor', 'left')
    .style('alignment-baseline', 'middle')
    .style('opacity', 0.8)
}
