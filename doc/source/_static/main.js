$(document).ready(function() {
  // Create radio buttons and insert them before the table
  var radioButtons = `
    <h3>Curvature</h3>
    <form id="filterForm">
      <label>
        <input type="radio" name="filter" value="all" checked> All
      </label>
      <label>
        <input type="radio" name="filter" value="convex"> Convex
      </label>
      <label>
        <input type="radio" name="filter" value="concave"> Concave
      </label>
    </form>
  `;

  $('table.scalar').before(radioButtons);

  // Initialize DataTable
  var table = $('table.scalar').DataTable();

  // Add event listener to radio buttons for filtering
  $('input[name="filter"]').change(function() {
    var selectedValue = $(this).val();

    if (selectedValue === "all") {
      table.search('').columns().search('').draw();
    } else {
      // Apply the search filter to the specific column (index 4) in the table
      table.column(4).search(selectedValue).draw();
    }
  });
      // Create a container div
    var containerDiv = $('<div>');

    var convexTable = `
      <h3>Table of only Convex Atoms</h3>
    `;
      // Create a new table after the original one
    var newTable = $('<table>').addClass('convex-view');

        // Get the filtered data from column 4 and extract the text from the <p> elements
    var filteredData = table.column(4).search("convex").data().toArray().map(function(cell) {
      var pElement = $(cell).find('p');
      return pElement.text();
    });
    // Add the filtered data to the new table
    newTable.DataTable({
      data: filteredData,
      columns: [
        { title: "Function" },
        { title: "Meaning" },
        { title: "Domain" },
        { title: "Sign" },
        { title: "Monotonicity" },
      ]
    });

    // Append both convexTable and newTable to the container
    containerDiv.append(convexTable);
    containerDiv.append(newTable);

    // Insert the container after the original table
    $('table.scalar').after(containerDiv);
});
