enroot exec $(enroot list -f | awk 'NR==2{print $2}') bash
