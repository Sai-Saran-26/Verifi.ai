
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import { Home } from './Home';
import { Works } from './Works';
import { Contactus } from './Contactus';
import { Dragdrop } from './Dragdrop';
import  {Results}  from './Results';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/works" element={<Works />} />
        <Route path="/contactus" element={<Contactus />} />
        <Route path="/dragdrop" element={<Dragdrop />} />
        <Route path="/results" element={<Results />} /> 
      </Routes>
    </Router>
  );
}

export default App;
