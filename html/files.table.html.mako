<%
# sort segments properly
def segment_key(args):
    seg1, seg2 = args
    return (seg1, -seg2)

skeys = sorted(prob_dirs.keys(), key=segment_key)
%>

<table class="hoverTable">
    <tr>
        <td>Data Range</td>
        <td>Model Probabilities</td>
        <td>Samples from Prior or Posterior</td>
    </tr>
% for dr in skeys:
    <tr>
        <td>${"{:,d} to {:,d}".format(int(dr[0]), int(dr[1]))}</td>
        <td><table>
    % for beta in sorted(betas):
          <tr>
            <td>beta: ${"{:.2f}".format(beta)} </td>
            <td><a href="mprobs/${prob_dirs[dr]}/${beta}/3">Top 3</a></td>
            <td><a href="mprobs/${prob_dirs[dr]}/${beta}/5">Top 5</a></td>
            <td><a href="mprobs/${prob_dirs[dr]}/${beta}/10">Top 10</a></td>
          </tr>
    % endfor
        </table></td>
        <td><table>
    % for beta in sorted(betas):
        <%
            sdr = (dr[0], dr[1], beta)
        %>
          <tr>
            <td><a href="hmuCmu_sampleplot/${sample_dirs[sdr]}">hmu-Cmu [beta: ${"{:.2f}".format(beta)}]</a></td>
          </tr>
    % endfor
        </table></td>
    </tr>
% endfor
</table>
