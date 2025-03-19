import React from 'react'
import './Navbar.css'
import { useNavigate } from 'react-router-dom';
export const Navbar = () => {
   const navigate = useNavigate(); 
  return (
    <div className='Nav-bar'>
        <div className='logo'>
            <span>Verifi.ai</span>
        </div>
        <div className='Nav-bar-list'>
                <span >Home</span>
                <span onClick={() => navigate('/contactus')}>Contact Us</span>
                <span onClick={() => navigate('/works')}>How it Works ?</span>
        </div>
        <div className='Nav-btn'>
            <button  onClick={() => navigate('/dragdrop')}>Try now</button>
        </div>
    </div>
  )
}
