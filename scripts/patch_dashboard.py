import io

path = "templates/dashboard.html"

with open(path, encoding="utf-8") as f:
    content = f.read()

# --- Edit 1: Download Report button ---
old1 = """<a href="{{ url_for('add_transaction') }}" class="btn btn-bank-success btn-sm">+ Add Transaction</a>"""
new1 = old1 + """
        <a href="{{ url_for('dashboard_report_pdf') }}" class="btn btn-bank-outline btn-sm" style="margin-left:8px;">Download Report</a>"""

count1 = content.count(old1)
print(f"Edit 1 (Download Report button) -- pattern found {count1} time(s)")
if count1 == 1:
    content = content.replace(old1, new1, 1)
    print("  -> applied")
elif count1 == 0:
    print("  -> NOT FOUND, skipped. Your file may already differ from what was expected.")
else:
    print("  -> multiple matches, skipped to avoid ambiguity. Needs manual review.")

# --- Edit 2: Severity breakdown chart ---
old2 = """        {% endif %}
      </div>
    </div>
  </div>
</div>"""

new2 = """        {% endif %}
      </div>
    </div>

    <div class="card-bank mt-4">
      <div class="card-bank-header">
        <h5>Alert Severity Breakdown</h5>
      </div>
      <div class="card-bank-body">
        <canvas id="severityChart" height="180"></canvas>
      </div>
    </div>
  </div>
</div>

<script>
new Chart(document.getElementById('severityChart').getContext('2d'), {
  type: 'doughnut',
  data: {
    labels: ['High', 'Medium', 'Low'],
    datasets: [{
      data: [
        {{ alerts|selectattr('severity','equalto','High')|list|length }},
        {{ alerts|selectattr('severity','equalto','Medium')|list|length }},
        {{ alerts|selectattr('severity','equalto','Low')|list|length }}
      ],
      backgroundColor: ['#dc2626', '#d97706', '#16a34a']
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: true, position: 'bottom' } }
  }
});
</script>"""

count2 = content.count(old2)
print(f"Edit 2 (severity chart) -- pattern found {count2} time(s)")
if count2 == 1:
    content = content.replace(old2, new2, 1)
    print("  -> applied")
elif count2 == 0:
    print("  -> NOT FOUND, skipped. Your file may already differ from what was expected.")
else:
    print("  -> multiple matches, skipped to avoid ambiguity. Needs manual review.")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done. dashboard.html has been updated in place.")
