<table class="hoverTable">
<tr>
  <td>Model Topology</td>
  <td>Log Evidence</td>
  <td>Probability</td>
  <td>Cumulative Probability</td>
</tr>
% for em, pr, cpr, evi in mi:
    <tr><td><a href="/mapmachine/${inferdir}/${em}">${em}</a><td>${evi['log_evidence']}</td></td><td>${pr}</td><td>${cpr}</td></tr>
% endfor
</table>
